from __future__ import annotations

import time
import sqlite3
import calendar
from datetime import datetime, timedelta, time as dtime
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from google import genai
from google.genai import types

# genai 버전 확인
try:
    import importlib.metadata as im
    genai_version = im.version("google-genai")
except Exception:
    genai_version = "버전 확인 불가"

# ==========================================
# 0.5 열람실 좌석 규칙
# ==========================================
SEAT_CLOSE_HOUR = 23  # 23:00
SEAT_OPEN_HOUR = 6    # 06:00
SEAT_BASE_MIN = 180   # 기본 3시간

SEAT_ALERT_WINDOW_SEC = 59 * 60  # 59분 이하부터 알림


def _dt_at(dt: datetime, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return dt.replace(hour=hh, minute=mm, second=ss, microsecond=0)


def is_seat_reset_window(now: datetime) -> bool:
    t = now.time()
    return (t >= dtime(SEAT_CLOSE_HOUR, 0)) or (t < dtime(SEAT_OPEN_HOUR, 0))


def next_seat_open_dt(now: datetime) -> datetime:
    today_open = _dt_at(now, SEAT_OPEN_HOUR, 0, 0)
    if now.time() < dtime(SEAT_OPEN_HOUR, 0):
        return today_open
    return today_open + timedelta(days=1)


def seat_close_dt_for(start_dt: datetime) -> datetime:
    return start_dt.replace(hour=SEAT_CLOSE_HOUR, minute=0, second=0, microsecond=0)


def get_seat_expiry_dt(seat_start_dt: datetime, extension_min: int) -> datetime:
    base_expiry = seat_start_dt + timedelta(minutes=SEAT_BASE_MIN + int(extension_min))
    close_dt = seat_close_dt_for(seat_start_dt)
    return min(base_expiry, close_dt)


def compute_seat_left_seconds(
    now: datetime, seat_start_dt: Optional[datetime], extension_min: int
) -> Optional[float]:
    if not seat_start_dt:
        return None
    if is_seat_reset_window(now):
        return None
    if now < seat_start_dt:
        return None
    expiry = get_seat_expiry_dt(seat_start_dt, extension_min)
    return (expiry - now).total_seconds()


def format_hms(sec: float) -> str:
    s = int(max(0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h}:{m:02d}:{ss:02d}"


# ==========================================
# 1. DB 유틸
# ==========================================
DB_PATH = "study_manager.db"


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    c = conn.cursor()

    # interruptions: phase 컬럼 포함
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS interruptions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            reason TEXT,
            duration_lost INTEGER DEFAULT 0,
            phase TEXT DEFAULT 'UNKNOWN'
        )
        """
    )
    # 기존 DB에서 phase 컬럼이 없던 경우를 위한 마이그레이션
    try:
        c.execute("SELECT phase FROM interruptions LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interruptions ADD COLUMN phase TEXT DEFAULT 'UNKNOWN'")

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            focus_minutes INTEGER
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY,
            task TEXT,
            status TEXT,
            date TEXT,
            is_subtask INTEGER,
            task_order INTEGER DEFAULT 999
        )
        """
    )
    try:
        c.execute("SELECT task_order FROM todos LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE todos ADD COLUMN task_order INTEGER DEFAULT 999")

    c.execute("UPDATE interruptions SET phase='UNKNOWN' WHERE phase IS NULL")
    conn.commit()


def reset_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS interruptions")
    c.execute("DROP TABLE IF EXISTS study_sessions")
    c.execute("DROP TABLE IF EXISTS todos")
    conn.commit()
    conn.close()
    st.cache_resource.clear()

def delete_records(table_name: str, id_list: list[int]) -> None:
    if not id_list:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 안전한 쿼리 생성을 위해 placeholder 사용
    placeholders = ', '.join('?' for _ in id_list)
    query = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"
    c.execute(query, id_list)
    conn.commit()
    conn.close()
    st.cache_resource.clear() # 캐시 초기화


# ==========================================
# 2. 학습 세션 로깅
# ==========================================
def _minutes_between(t0: datetime, t1: datetime) -> int:
    sec = max(0, int((t1 - t0).total_seconds()))
    return sec // 60


def log_focus_segment_if_any(conn: sqlite3.Connection, seg_start: Optional[datetime], seg_end: datetime) -> int:
    if not seg_start:
        return 0
    mins = _minutes_between(seg_start, seg_end)
    if mins <= 0:
        return 0
    c = conn.cursor()
    c.execute(
        "INSERT INTO study_sessions (start_time, end_time, focus_minutes) VALUES (?, ?, ?)",
        (seg_start.strftime("%Y-%m-%d %H:%M:%S"), seg_end.strftime("%Y-%m-%d %H:%M:%S"), mins),
    )
    conn.commit()
    return mins


# ==========================================
# 3. Gemini 리포트
# ==========================================
def _time_band(h: int) -> str:
    if h < 6:
        return "새벽(00-06)"
    if h < 12:
        return "오전(06-12)"
    if h < 18:
        return "오후(12-18)"
    return "저녁(18-24)"


def ai_generate_report(api_key: str, df_focus: pd.DataFrame, df_interrupt: pd.DataFrame, period_label: str, days: int) -> str:
    if not api_key:
        return "API 키가 입력되지 않았습니다."

    cutoff = datetime.now() - timedelta(days=days)

    # Focus summary
    f = df_focus.copy()
    if "start_time" not in f.columns:
        return "학습 데이터 형식이 올바르지 않습니다."
    f["start_time"] = pd.to_datetime(f["start_time"], errors="coerce")
    f["focus_minutes"] = pd.to_numeric(f.get("focus_minutes"), errors="coerce").fillna(0)
    f = f.dropna(subset=["start_time"])
    f = f[f["start_time"] >= cutoff].copy()

    total_min = int(f["focus_minutes"].sum())
    total_hr = round(total_min / 60.0, 1)

    f["date"] = f["start_time"].dt.date
    daily = f.groupby("date", as_index=False)["focus_minutes"].sum().sort_values("date")
    active_days = int((daily["focus_minutes"] > 0).sum()) if not daily.empty else 0
    avg_daily = int(total_min / max(1, active_days))

    trend_text = "데이터가 아직 부족합니다."
    if len(daily) >= 4:
        last3 = int(daily.tail(3)["focus_minutes"].sum())
        prev = int(daily.iloc[:-3]["focus_minutes"].sum())
        prev_days = max(1, len(daily) - 3)
        prev3_scaled = int(prev / prev_days * 3)
        delta = last3 - prev3_scaled
        if delta >= 30:
            trend_text = f"최근 3일이 이전 평균(3일 환산)보다 약 {delta}분 더 많아 상승 흐름이 보입니다."
        elif delta <= -30:
            trend_text = f"최근 3일이 이전 평균(3일 환산)보다 약 {abs(delta)}분 줄어 잠깐 주춤한 흐름입니다."
        else:
            trend_text = "최근 3일과 이전 평균이 비슷해 안정적인 흐름입니다."

    best_day = "없음"
    worst_day = "없음"
    if not daily.empty:
        best = daily.loc[daily["focus_minutes"].idxmax()]
        worst = daily.loc[daily["focus_minutes"].idxmin()]
        best_day = f"{best['date']}에 {int(best['focus_minutes'])}분"
        worst_day = f"{worst['date']}에 {int(worst['focus_minutes'])}분"

    rhythm_weekday = "데이터 부족"
    rhythm_band = "데이터 부족"
    if not f.empty:
        f["weekday"] = f["start_time"].dt.day_name()
        f["hour"] = f["start_time"].dt.hour
        f["time_band"] = f["hour"].apply(lambda x: _time_band(int(x)))

        wk = f.groupby("weekday")["focus_minutes"].sum().sort_values(ascending=False)
        bd = f.groupby("time_band")["focus_minutes"].sum().sort_values(ascending=False)

        if not wk.empty:
            rhythm_weekday = f"{wk.index[0]} ({int(wk.iloc[0])}분)"
        if not bd.empty:
            rhythm_band = f"{bd.index[0]} ({int(bd.iloc[0])}분)"

    it = df_interrupt.copy()
    if "timestamp" in it.columns:
        it["timestamp"] = pd.to_datetime(it["timestamp"], errors="coerce")
        it = it.dropna(subset=["timestamp"])
        it = it[it["timestamp"] >= cutoff].copy()
    else:
        it = it.iloc[0:0]

    # AI 리포트에서도 FOCUS 중 기록만 집계
    it_focus = it[it.get("phase", "UNKNOWN") == "FOCUS"] if not it.empty else it

    interrupt_cnt = int(len(it_focus))
    top_interrupt = "중단 기록이 없습니다."
    biggest_one = "없음"
    if not it_focus.empty and "reason" in it_focus.columns:
        vc = it_focus["reason"].value_counts()
        top3 = vc.head(3)
        top_interrupt = "\n".join([f"- {k}: {int(v)}회" for k, v in top3.items()])
        biggest_one = str(top3.index[0])

    prompt = f"""
너는 따뜻하지만 날카로운 '학습 코치'다.
아래 데이터를 바탕으로 사용자가 “읽고 바로 행동할 수 있는” 상세 리포트를 한국어로 작성해라.

[기간]
- {period_label} (최근 {days}일)

[집중 요약]
- 총 집중 시간: {total_min}분 (약 {total_hr}시간)
- 실제 공부한 날(집중 1분 이상): {active_days}일
- 공부한 날 기준 하루 평균: 약 {avg_daily}분
- 가장 집중한 날: {best_day}
- 가장 집중이 적었던 날: {worst_day}
- 흐름(트렌드): {trend_text}

[집중 패턴]
- 가장 집중이 잘 된 요일: {rhythm_weekday}
- 가장 집중이 잘 된 시간대: {rhythm_band}

[중단/방해]  (※ '집중(FOCUS) 중' 기록만 집계)
- 중단/종료 발생: {interrupt_cnt}회
- 상위 방해 요인:
{top_interrupt}
- 가장 큰 방해 요인(최빈): {biggest_one}

[작성 규칙(중요)]
- 절대 '1.' '2)' '•' 같은 번호/목록 형식을 쓰지 말고, 자연스러운 서술형 문단 4~6개로 작성해라.
- 문단 구성 가이드:
  첫 문단: 기간 전체를 한 문장으로 요약 + 사용자를 인정/격려.
  둘째 문단: 집중량(총량/평균/베스트데이/워스트데이)을 해석해서 “왜 의미 있는지” 설명.
  셋째 문단: 집중 리듬(요일/시간대)을 해석하고, 사용자에게 맞는 공부 전략(언제 어떤 과제를 배치할지)로 연결.
  넷째 문단: 방해 요인을 기반으로 가장 큰 원인 1개를 콕 집어, 현실적인 해결 방법(환경/규칙/트리거 제거)을 제시.
  다섯째 문단: 내일 바로 실행할 “구체적 플랜”을 문장 속에 자연스럽게 포함(예: 언제, 무엇을, 얼마나).
  마지막 문단: 짧고 강한 동기부여 문장으로 마무리.
- 너무 일반론(‘꾸준히 해요’만) 금지. 반드시 위 수치와 패턴을 언급하며 구체적으로 써라.
- 전체 길이 900자 이내.
""".strip()

    try:
        client = genai.Client(api_key=api_key)
        model_id = "gemini-2.5-flash"
        resp = client.models.generate_content(
            model=model_id,
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(temperature=0.35, top_p=0.95),
        )
        text = getattr(resp, "text", None)
        return (text or "").strip() or "응답이 비어 있습니다."
    except Exception as e:
        return f"AI 리포트 생성 오류:\n{str(e)}"


# ==========================================
# 4. 타이머 원형 HTML
# ==========================================
def get_filled_pie_html(percentage: float, color: str, time_text: str, status_text: str) -> str:
    radius = 25
    circumference = 2 * 3.14159 * radius
    pct = max(0.0, min(100.0, float(percentage)))
    stroke_dasharray = f"{circumference * pct / 100.0} {circumference}"

    svg_html = f"""
    <div style="position: relative; width: 300px; height: 300px; margin-bottom: 10px;">
        <svg width="300" height="300" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="{radius}" fill="none" stroke="#eee" stroke-width="50" />
            <circle cx="50" cy="50" r="{radius}" fill="none" stroke="{color}" stroke-width="50"
                stroke-dasharray="{stroke_dasharray}"
                stroke-linecap="butt"
                transform="rotate(-90 50 50)"
                style="transition: stroke-dasharray 1s linear;"
            />
        </svg>
    </div>
    """
    text_html = f"""
    <div style="text-align: center;">
        <div style="font-size: 3.5rem; font-weight: bold; color: #333; line-height: 1.0;">{time_text}</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {color}; margin-top: 5px;">{status_text}</div>
    </div>
    """
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        {svg_html}
        {text_html}
    </div>
    """


# ==========================================
# 5. UI 및 로직
# ==========================================
st.set_page_config(page_title="AI Study Manager", layout="wide")

st.markdown(
    """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem !important; font-weight: 800 !important;
    }
    .stButton button { width: 100%; font-weight: bold; border-radius: 12px; }
    div[data-testid="column"] button {
        height: 44px !important; min-height: 44px !important; font-size: 1.0rem !important;
        padding: 0 1rem !important; margin-top: 0px !important; border-radius: 10px !important;
        border: 1px solid #ddd; line-height: 1 !important;
    }
    div[data-testid="stTextInput"] input { height: 44px !important; min-height: 44px !important; }
    .todo-text {
        height: 44px; display: flex; align-items: center; padding: 0 14px;
        border-radius: 10px; background: #f3f4f6; font-size: 1.0rem; width: 100%; margin-bottom: 6px;
    }
    .todo-done { color: #999; text-decoration: line-through; }
    .timer-title { font-size: 2rem; font-weight: 900; text-align: center; color: #333; margin-bottom: 10px; }
    .seat-box { text-align:center; margin-top:6px; padding:10px 12px; border:1px solid #e5e7eb; border-radius:12px; background:#fafafa; display: flex; align-items: center; justify-content: center; gap: 10px;}
    .seat-ok { color:#555; font-size:1.05rem; flex-grow: 1;}
    .seat-exp { color:#d33; font-size:1.05rem; font-weight:800; flex-grow: 1;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Session State defaults
# ----------------------------
defaults = {
    "running": False,
    "paused": False,
    "phase": "IDLE",
    "phase_start_dt": None,
    "phase_end_dt": None,
    "pause_started_at": None,
    "pause_snapshot_prog": None,
    "pause_snapshot_rem_sec": None,
    "seat_extension_min": 0,
    "show_extension_dialog": False,
    "extension_seat_left_sec": None,
    "seat_extension_context": "break",
    "seat_alert_shown_in_rest": False,
    "show_start_setup": False,
    "pending_start": False,
    "pending_resume": False,
    "pending_focus": 25,
    "pending_rest": 5,
    "show_pause_dialog": False,
    "show_stop_dialog": False,
    "prev_seat_toggle": False,
    "show_seat_settings": False,
    "seat_autopopup_done": False,
    "block_next_focus_until_seat_extended": False,
    "need_main_rerun": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "use_seat": False,
        "seat_start_dt": datetime.now().replace(second=0, microsecond=0),
        "focus": 25,
        "rest": 5,
    }

conn = get_conn()
init_db(conn)


def _clear_dialog_flags() -> None:
    st.session_state["show_start_setup"] = False
    st.session_state["show_extension_dialog"] = False
    st.session_state["show_pause_dialog"] = False
    st.session_state["show_stop_dialog"] = False
    st.session_state["show_seat_settings"] = False


def _open_dialog(name: str) -> None:
    _clear_dialog_flags()
    st.session_state[name] = True
    st.rerun()


def _request_extension_popup(context: str, seat_left_sec: float) -> None:
    st.session_state["seat_extension_context"] = context
    st.session_state["extension_seat_left_sec"] = float(max(0.0, seat_left_sec))
    _clear_dialog_flags()
    st.session_state["show_extension_dialog"] = True
    st.session_state["need_main_rerun"] = True


def _start_timer_session(now: datetime, focus_min: int, rest_min: int) -> None:
    st.session_state["settings"]["focus"] = int(focus_min)
    st.session_state["settings"]["rest"] = int(rest_min)

    st.session_state["running"] = True
    st.session_state["paused"] = False
    st.session_state["pause_started_at"] = None
    st.session_state["pause_snapshot_prog"] = None
    st.session_state["pause_snapshot_rem_sec"] = None

    st.session_state["phase"] = "FOCUS"
    st.session_state["phase_start_dt"] = now
    st.session_state["phase_end_dt"] = now + timedelta(minutes=int(focus_min))

    st.session_state["seat_alert_shown_in_rest"] = False
    st.session_state["seat_extension_context"] = "break"
    st.session_state["seat_extension_min"] = int(st.session_state.get("seat_extension_min", 0))

    st.session_state["block_next_focus_until_seat_extended"] = False

    st.success("🚀 학습을 시작합니다!")


def _switch_phase(now: datetime, to_phase: str) -> None:
    focus_min = int(st.session_state["settings"].get("focus", 25))
    rest_min = int(st.session_state["settings"].get("rest", 5))

    st.session_state["phase"] = to_phase
    st.session_state["phase_start_dt"] = now

    if to_phase == "FOCUS":
        st.session_state["phase_end_dt"] = now + timedelta(minutes=focus_min)
        st.session_state["phase_start_dt"] = now

    elif to_phase == "REST":
        st.session_state["phase_end_dt"] = now + timedelta(minutes=rest_min)
        st.session_state["phase_start_dt"] = now

        st.session_state["seat_alert_shown_in_rest"] = False

        if st.session_state["settings"].get("use_seat", False) and (not is_seat_reset_window(now)):
            seat_start_dt = st.session_state["settings"].get("seat_start_dt")
            seat_left_sec = compute_seat_left_seconds(
                now, seat_start_dt, st.session_state.get("seat_extension_min", 0)
            )
            if (seat_left_sec is not None) and (seat_left_sec <= SEAT_ALERT_WINDOW_SEC):
                st.session_state["seat_alert_shown_in_rest"] = True
                _request_extension_popup(context="break_start", seat_left_sec=float(seat_left_sec))

    else:
        st.session_state["phase_end_dt"] = None


def _resume_timer(now: datetime) -> None:
    pause_started_at = st.session_state.get("pause_started_at")
    if pause_started_at:
        paused_delta = now - pause_started_at
        ps = st.session_state.get("phase_start_dt")
        pe = st.session_state.get("phase_end_dt")
        if ps is not None:
            st.session_state["phase_start_dt"] = ps + paused_delta
        if pe is not None:
            st.session_state["phase_end_dt"] = pe + paused_delta

    st.session_state["paused"] = False
    st.session_state["pause_started_at"] = None
    st.session_state["pause_snapshot_prog"] = None
    st.session_state["pause_snapshot_rem_sec"] = None

    st.success("▶️ 학습을 재개합니다!")


# ==========================================
# Dialogs
# ==========================================
@st.dialog("🪑 좌석시간 설정")
def seat_settings_dialog():
    st.subheader("좌석/예약 관리")
    st.caption("학교 규칙 반영: 21시에 예약해도 만료는 23:00, 23:00~06:00은 예약 불필요.")

    time_ref = st.radio("기준 선택", ["예약 시작 시간", "예약 만료 시간"], horizontal=True)
    input_method = st.radio("입력 방식", ["시계로 선택", "직접 입력"], horizontal=True)

    now = datetime.now().replace(second=0, microsecond=0)
    current_dt: datetime = st.session_state["settings"].get("seat_start_dt", now)
    current_time = current_dt.time()

    new_time: Optional[dtime] = None
    if input_method == "시계로 선택":
        new_time = st.time_input("시간 선택", value=current_time, step=60)
    else:
        time_str = st.text_input("시간 입력 (예: 14:00 또는 1400)", value=current_time.strftime("%H:%M"))
        try:
            if ":" in time_str:
                new_time = datetime.strptime(time_str, "%H:%M").time()
            elif len(time_str) == 4:
                new_time = datetime.strptime(time_str, "%H%M").time()
        except Exception:
            new_time = None

    st.write("")
    if st.button("저장", type="primary", width="stretch"):
        if new_time:
            if time_ref == "예약 시작 시간":
                candidate = datetime.combine(now.date(), new_time)
                if candidate > now + timedelta(minutes=5):
                    candidate -= timedelta(days=1)
                st.session_state["settings"]["seat_start_dt"] = candidate
            else:
                expiry_time = new_time
                if expiry_time > dtime(SEAT_CLOSE_HOUR, 0):
                    expiry_time = dtime(SEAT_CLOSE_HOUR, 0)
                expiry_candidate = datetime.combine(now.date(), expiry_time)
                if expiry_candidate < now - timedelta(minutes=5):
                    expiry_candidate += timedelta(days=1)
                seat_start = expiry_candidate - timedelta(minutes=SEAT_BASE_MIN)
                st.session_state["settings"]["seat_start_dt"] = seat_start

            st.session_state["seat_extension_min"] = 0
            st.session_state["seat_alert_shown_in_rest"] = False
            st.session_state["extension_seat_left_sec"] = None
            st.session_state["block_next_focus_until_seat_extended"] = False

        st.success("좌석 시간 저장 완료")
        time.sleep(0.5)
        _clear_dialog_flags()
        st.rerun()

    if st.button("닫기", width="stretch"):
        _clear_dialog_flags()
        st.rerun()


@st.dialog("🚀 공부 시작 설정")
def start_setup_dialog():
    st.subheader("학습 모드 선택")
    mode_options = ["25분 집중 / 5분 휴식", "50분 집중 / 10분 휴식", "테스트 모드 (2분 집중 / 1분 휴식)"]
    current_focus = st.session_state["settings"].get("focus", 25)

    default_idx = 0
    if current_focus == 50:
        default_idx = 1
    elif current_focus == 2:
        default_idx = 2
    mode = st.radio("타이머 모드", mode_options, index=default_idx)

    st.write("")
    if st.button("시작하기", type="primary", width="stretch"):
        if "테스트" in mode:
            f, r = 2, 1
        elif "25분" in mode:
            f, r = 25, 5
        else:
            f, r = 50, 10

        now = datetime.now().replace(microsecond=0)

        if st.session_state["settings"].get("use_seat", False) and (not is_seat_reset_window(now)):
            seat_start_dt = st.session_state["settings"].get("seat_start_dt")
            
            left_sec = compute_seat_left_seconds(now, seat_start_dt, st.session_state.get("seat_extension_min", 0))
            
            if (left_sec is not None) and (left_sec <= SEAT_ALERT_WINDOW_SEC):
                st.session_state["pending_start"] = True
                st.session_state["pending_focus"] = f
                st.session_state["pending_rest"] = r
                
                _request_extension_popup(context="prestart", seat_left_sec=float(left_sec))
                st.rerun()
                return

        _clear_dialog_flags()
        _start_timer_session(now, f, r)
        time.sleep(0.5)
        st.rerun()

    if st.button("닫기", width="stretch"):
        _clear_dialog_flags()
        st.rerun()

@st.dialog("🚨 좌석 체크")
def extension_dialog():
    ctx = st.session_state.get("seat_extension_context", "break")
    seat_left_sec = st.session_state.get("extension_seat_left_sec") or 0
    left_min = int(seat_left_sec // 60)

    # 1. 만료 여부 확인
    is_expired = (seat_left_sec <= 0)

    # 2. 메시지 표시
    if is_expired:
        st.error("⚠️ 좌석 이용 시간이 끝났습니다!", icon="🚫")
        st.warning("규칙에 따라 학습을 진행할 수 없습니다. 좌석을 다시 예약한 후 아래 버튼으로 시간 정보를 갱신하세요.")
    else:
        # 임박 상태 (59분 이하)
        if ctx == "prestart":
            st.info(f"시작하려면 좌석 연장이 필요합니다. (남은 시간: {left_min}분)")
        elif ctx == "resume":
            st.info(f"재개하려면 좌석 연장이 필요합니다. (남은 시간: {left_min}분)")
        else:
            # 휴식 -> 집중 차단 시
            st.warning(f"좌석 시간이 부족하여 다음 집중을 시작할 수 없습니다. ({left_min}분 남음)")

    # 3. 연장 버튼 (만료 여부 관계없이 띄우거나, 정책에 따라 만료 시 숨김 가능)
    changed = False
    if not is_expired:
        c1, c2, c3 = st.columns(3)
        if c1.button("1시간", width="stretch"):
            st.session_state["seat_extension_min"] += 60
            changed = True
        if c2.button("2시간", width="stretch"):
            st.session_state["seat_extension_min"] += 120
            changed = True
        if c3.button("3시간", width="stretch"):
            st.session_state["seat_extension_min"] += 180
            changed = True

    # 4. 연장 성공 시 -> 차단 풀고 진행
    if changed:
        _clear_dialog_flags()
        st.session_state["extension_seat_left_sec"] = None
        
        now = datetime.now().replace(microsecond=0)
        # 차단 해제 및 상태 복구
        if st.session_state.get("block_next_focus_until_seat_extended", False):
            st.session_state["block_next_focus_until_seat_extended"] = False
            _switch_phase(now, "FOCUS")
            st.rerun()
            
        # 대기 중이던 시작/재개 실행
        if st.session_state.get("pending_start", False):
            st.session_state["pending_start"] = False
            _start_timer_session(now, int(st.session_state["pending_focus"]), int(st.session_state["pending_rest"]))
            st.rerun()
            
        if st.session_state.get("pending_resume", False):
            st.session_state["pending_resume"] = False
            _resume_timer(now)
            st.rerun()
        st.rerun()

    # -------------------------------------------------------
    # 5. 닫기 버튼의 UX 변경
    # -------------------------------------------------------    
    if is_expired:
        # 만료됨 -> 확인 누르면 종료
        close_label = "확인 (학습 종료)"
    elif ctx in ["prestart", "resume"]:
        # 시작/재개 전 -> 취소 누르면 시작 안 함
        close_label = "취소 (시작 안 함)"
    else:
        # 휴식 중 차단 -> 닫기 누르면 학습 종료
        close_label = "종료 (그만하기)"
    
    if st.button(close_label, width="stretch"):
        _clear_dialog_flags()
        
        # [Strict Mode] 연장 없이 닫으면 -> 작업을 취소하거나 세션을 종료함
        
        # 1. 차단 상태였거나 만료 상태였으면 -> 타이머 강제 종료
        if st.session_state.get("block_next_focus_until_seat_extended", False) or is_expired:
             st.session_state["block_next_focus_until_seat_extended"] = False
             st.session_state["running"] = False 
             st.session_state["phase"] = "IDLE"
             
        # 2. 시작/재개 대기 중이었으면 -> 요청 취소
        else:
            st.session_state["pending_start"] = False
            st.session_state["pending_resume"] = False
             
        st.rerun()

PAUSE_REASONS = ["화장실", "물/커피", "연락/전화", "SNS/핸드폰", "주변 소음/방해", "기타"]
STOP_REASONS = ["공부 끝(목표 달성)", "다음 일정/수업", "피로/졸림", "집중 안 됨(컨디션)", "급한 일 생김", "기타"]


@st.dialog("⏸️ 중단(일시정지)")
def pause_dialog():
    st.write("타이머를 **일시정지**합니다.")
    reason = st.selectbox("중단 사유", PAUSE_REASONS)

    if st.button("일시정지", type="primary", width="stretch"):
        now = datetime.now().replace(microsecond=0)
        cur_phase = str(st.session_state.get("phase", "UNKNOWN"))

        if cur_phase == "FOCUS" and st.session_state.get("phase_start_dt"):
            log_focus_segment_if_any(conn, st.session_state.get("phase_start_dt"), now)

        conn.cursor().execute(
            "INSERT INTO interruptions (timestamp, reason, duration_lost, phase) VALUES (?, ?, ?, ?)",
            (now.strftime("%Y-%m-%d %H:%M:%S"), f"[중단] {reason}", 0, cur_phase),
        )
        conn.commit()

        phase_start = st.session_state.get("phase_start_dt")
        phase_end = st.session_state.get("phase_end_dt")
        if cur_phase in ("FOCUS", "REST") and phase_start and phase_end:
            total_sec = max(1.0, (phase_end - phase_start).total_seconds())
            rem_sec = max(0.0, (phase_end - now).total_seconds())
            elapsed = max(0.0, (now - phase_start).total_seconds())
            prog = min(100.0, (elapsed / total_sec) * 100.0)
            st.session_state["pause_snapshot_prog"] = float(prog)
            st.session_state["pause_snapshot_rem_sec"] = float(rem_sec)
        else:
            st.session_state["pause_snapshot_prog"] = 0.0
            st.session_state["pause_snapshot_rem_sec"] = 0.0

        st.session_state["paused"] = True
        st.session_state["pause_started_at"] = now

        st.success("⏸️ 일시정지 완료!")
        time.sleep(0.5)
        _clear_dialog_flags()
        st.rerun()

    if st.button("닫기", width="stretch"):
        _clear_dialog_flags()
        st.rerun()


@st.dialog("🏁 종료")
def stop_dialog():
    st.write("세션을 **종료**합니다.")
    reason = st.selectbox("종료 사유", STOP_REASONS)

    if st.button("종료하기", type="primary", width="stretch"):
        now = datetime.now().replace(microsecond=0)
        cur_phase = str(st.session_state.get("phase", "UNKNOWN"))

        if cur_phase == "FOCUS" and st.session_state.get("phase_start_dt"):
            log_focus_segment_if_any(conn, st.session_state.get("phase_start_dt"), now)

        conn.cursor().execute(
            "INSERT INTO interruptions (timestamp, reason, duration_lost, phase) VALUES (?, ?, ?, ?)",
            (now.strftime("%Y-%m-%d %H:%M:%S"), f"[종료] {reason}", 0, cur_phase),
        )
        conn.commit()

        st.session_state["running"] = False
        st.session_state["paused"] = False
        st.session_state["pause_started_at"] = None
        st.session_state["pause_snapshot_prog"] = None
        st.session_state["pause_snapshot_rem_sec"] = None
        st.session_state["phase"] = "IDLE"
        st.session_state["phase_start_dt"] = None
        st.session_state["phase_end_dt"] = None
        st.session_state["seat_alert_shown_in_rest"] = False
        st.session_state["block_next_focus_until_seat_extended"] = False

        st.success("학습 종료!")
        time.sleep(0.5)
        _clear_dialog_flags()
        st.rerun()

    if st.button("닫기", width="stretch"):
        _clear_dialog_flags()
        st.rerun()


# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    st.info(f"📚 google-genai 버전: {genai_version}")
    api_key = st.text_input("Gemini API Key", type="password")

    if st.button("🔑 API 키 테스트"):
        if not api_key:
            st.error("API 키를 먼저 입력해주세요.")
        else:
            try:
                _ = genai.Client(api_key=api_key)
                st.success("✅ 연결 성공!")
            except Exception as e:
                st.error(f"❌ 연결 실패:\n{str(e)}")

    st.divider()
    if st.button("🗑️ 데이터 초기화", width="stretch"):
        reset_db()
        st.success("데이터 삭제 완료!")
        time.sleep(0.5)
        st.rerun()


# ==========================================
# Main Tabs
# ==========================================
tab1, tab2 = st.tabs(["⏱️ 타이머 & To-Do", "📊 리포트"])

with tab1:
    col_timer, col_todo = st.columns([1, 1])

    with col_timer:
        st.markdown('<div class="timer-title">Study Timer</div>', unsafe_allow_html=True)
        lock_settings = st.session_state["running"] and (not st.session_state["paused"])

        topA, topB = st.columns([3, 3], gap="small")

        with topA:
            seat_col_toggle, seat_col_btn = st.columns([4, 1], gap="small")
            with seat_col_toggle:
                seat_toggle = st.toggle(
                    "🪑 좌석 예약",
                    value=st.session_state["settings"].get("use_seat", False),
                    disabled=lock_settings,
                )

            want_open_seat_dialog = False
            with seat_col_btn:
                if seat_toggle:
                    if st.button("⚙️", key="seat_edit_top", disabled=lock_settings, width="stretch"):
                        if not lock_settings:
                            want_open_seat_dialog = True

            prev = st.session_state.get("prev_seat_toggle", False)

            # use_seat / prev 업데이트
            st.session_state["settings"]["use_seat"] = seat_toggle
            st.session_state["prev_seat_toggle"] = seat_toggle

            # 토글 OFF 되면 자동팝업 다시 가능하도록 리셋
            if prev and (not seat_toggle):
                st.session_state["seat_autopopup_done"] = False

            if (not prev) and seat_toggle:
                saved_dt = st.session_state["settings"].get("seat_start_dt")
                now_date = datetime.now().date()
                
                if saved_dt and saved_dt.date() < now_date:
                     st.session_state["settings"]["seat_start_dt"] = datetime.now().replace(second=0, microsecond=0)
                
                if (not lock_settings) and (not st.session_state.get("seat_autopopup_done", False)):
                    st.session_state["seat_autopopup_done"] = True
                    want_open_seat_dialog = True

            if want_open_seat_dialog:
                _open_dialog("show_seat_settings")

        with topB:
            if not st.session_state["running"]:
                if st.button("▶️ 공부 시작", type="primary", width="stretch"):
                    _open_dialog("show_start_setup")
            else:
                if st.session_state["paused"]:
                    if st.button("▶️ 재개", type="primary", width="stretch"):
                        now = datetime.now().replace(microsecond=0)
                        
                        # 플래그 변수 미리 초기화
                        is_seat_issue = False 

                        if st.session_state["settings"].get("use_seat", False) and (not is_seat_reset_window(now)):
                            seat_start_dt = st.session_state["settings"].get("seat_start_dt")
                            left_sec = compute_seat_left_seconds(
                                now, seat_start_dt, st.session_state.get("seat_extension_min", 0)
                            )
                            
                            # 좌석 문제 발생 시 플래그를 True로 설정
                            if (left_sec is not None) and (left_sec <= SEAT_ALERT_WINDOW_SEC):
                                is_seat_issue = True  
                                st.session_state["pending_resume"] = True
                                st.session_state["seat_extension_context"] = "resume"
                                st.session_state["extension_seat_left_sec"] = float(left_sec) 
                                _open_dialog("show_extension_dialog")
                        
                        # 문제가 없을 때만(False일 때만) 타이머 재개
                        if not is_seat_issue:
                            _resume_timer(now)
                            time.sleep(0.5)
                            st.rerun()
                else:
                    st.button("⏱️ 실행 중", width="stretch", disabled=True)

        @st.fragment(run_every=1)
        def seat_always_box():
            if not st.session_state["settings"].get("use_seat", False):
                return
            now = datetime.now().replace(microsecond=0)
            if is_seat_reset_window(now):
                nxt = next_seat_open_dt(now)
                msg = f"🪑 23:00~06:00 (예약 불필요) · 다음 운영 {nxt.strftime('%H:%M')}"
                st.markdown(
                    f"<div class='seat-box' style='margin-top:0;'><div class='seat-ok'><b>{msg}</b></div></div>",
                    unsafe_allow_html=True,
                )
                return

            seat_start_dt = st.session_state["settings"].get("seat_start_dt")
            if not seat_start_dt:
                st.markdown(
                    "<div class='seat-box' style='margin-top:0;'><div class='seat-exp'>🪑 좌석 시간 미설정</div></div>",
                    unsafe_allow_html=True,
                )
                return

            left_sec = compute_seat_left_seconds(now, seat_start_dt, st.session_state.get("seat_extension_min", 0))
            expiry = get_seat_expiry_dt(seat_start_dt, st.session_state.get("seat_extension_min", 0))

            if now < seat_start_dt:
                st.markdown(
                    f"<div class='seat-box' style='margin-top:0;'><div class='seat-ok'>🪑 예약 전 · 시작 {seat_start_dt.strftime('%H:%M')}</div></div>",
                    unsafe_allow_html=True,
                )
            elif left_sec is not None and left_sec > 0:
                txt = format_hms(left_sec)
                st.markdown(
                    f"<div class='seat-box' style='margin-top:0;'><div class='seat-ok'>🪑 만료까지 <b>{txt}</b> · ({expiry.strftime('%H:%M')})</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='seat-box' style='margin-top:0;'><div class='seat-exp'>🪑 좌석 만료</div></div>",
                    unsafe_allow_html=True,
                )

        seat_always_box()
        st.write("")

        @st.fragment(run_every=1)
        def run_timer_fragment():
            if not st.session_state["running"]:
                st.markdown(get_filled_pie_html(0, "#ccc", "00:00", "대기 중"), unsafe_allow_html=True)
                return

            now = datetime.now().replace(microsecond=0)

            # ---------------------------------------------------------
            # 1. 차단 상태 확인 (휴식 종료 후 좌석 연장 대기 중)
            # ---------------------------------------------------------
            if st.session_state.get("block_next_focus_until_seat_extended", False):
                # 강제로 상태를 REST로 고정하고, 시간을 멈춤
                st.session_state["phase"] = "REST"
                st.session_state["phase_end_dt"] = now  # 끝난 상태 유지
                
                # 팝업이 닫혀있다면 다시 켬 (연장할 때까지 팝업)
                if not st.session_state.get("show_extension_dialog", False):
                    # 다시 팝업 요청
                    st.session_state["seat_extension_context"] = "break"
                    if st.session_state["settings"].get("use_seat", False):
                        seat_start_dt = st.session_state["settings"].get("seat_start_dt")
                        seat_left_sec = compute_seat_left_seconds(
                            now, seat_start_dt, st.session_state.get("seat_extension_min", 0)
                        )
                        if seat_left_sec is not None:
                            st.session_state["extension_seat_left_sec"] = float(max(0.0, seat_left_sec))
                    
                    st.session_state["show_extension_dialog"] = True
                    st.rerun()

                # 화면 표시: 꽉 찬 초록색 원 + 00:00 + 대기 문구
                st.markdown(get_filled_pie_html(100, "#4CAF50", "00:00", "휴식(대기) ⛔"), unsafe_allow_html=True)
                return

            # ---------------------------------------------------------
            # 2. 일시정지 화면
            # ---------------------------------------------------------
            if st.session_state.get("paused"):
                phase = st.session_state.get("phase", "IDLE")
                is_focus = (phase == "FOCUS")
                color, status = ("#FF4B4B", "집중(일시정지) ⏸️") if is_focus else ("#4CAF50", "휴식(일시정지) ⏸️")
                snap_prog = st.session_state.get("pause_snapshot_prog")
                snap_rem = st.session_state.get("pause_snapshot_rem_sec")
                if snap_prog is None or snap_rem is None:
                    st.markdown(get_filled_pie_html(0, "#999", "PAUSE", "일시정지 ⏸️"), unsafe_allow_html=True)
                    return
                rem_sec = max(0.0, float(snap_rem))
                time_txt = f"{int(rem_sec//60):02d}:{int(rem_sec%60):02d}"
                st.markdown(get_filled_pie_html(float(snap_prog), color, time_txt, status), unsafe_allow_html=True)
                return

            # ---------------------------------------------------------
            # 3. 타이머 실행 화면
            # ---------------------------------------------------------
            phase = st.session_state.get("phase", "IDLE")
            phase_start = st.session_state.get("phase_start_dt")
            phase_end = st.session_state.get("phase_end_dt")
            
            if phase not in ("FOCUS", "REST") or (phase_start is None) or (phase_end is None):
                st.markdown(get_filled_pie_html(0, "#ccc", "00:00", "대기 중"), unsafe_allow_html=True)
                return

            total_sec = max(1.0, (phase_end - phase_start).total_seconds())
            rem_sec = max(0.0, (phase_end - now).total_seconds())
            elapsed = max(0.0, (now - phase_start).total_seconds())
            prog = min(100.0, (elapsed / total_sec) * 100.0)

            is_focus = (phase == "FOCUS")
            color, status = ("#FF4B4B", "집중 🔥") if is_focus else ("#4CAF50", "휴식 ☕")
            
            st.markdown(
                get_filled_pie_html(prog, color, f"{int(rem_sec//60):02d}:{int(rem_sec%60):02d}", status),
                unsafe_allow_html=True,
            )

            # ---------------------------------------------------------
            # 4. 구간 종료 처리 (0초 도달 시)
            # ---------------------------------------------------------
            if now >= phase_end:
                # [CASE A] FOCUS 종료 → REST 시작
                if phase == "FOCUS":
                    if st.session_state.get("phase_start_dt"):
                        log_focus_segment_if_any(conn, st.session_state.get("phase_start_dt"), now)

                    _switch_phase(now, "REST")
                    st.session_state["block_next_focus_until_seat_extended"] = False
                    
                    # 휴식 시작 팝업이 떴다면 즉시 띄우기 위해 리런
                    if st.session_state.get("show_extension_dialog", False):
                        st.rerun()
                    return

                # [CASE B] REST 종료 → FOCUS 넘어가기 "직전" 검사
                if phase == "REST":
                    use_seat = st.session_state["settings"].get("use_seat", False)

                    if use_seat and (not is_seat_reset_window(now)):
                        seat_start_dt = st.session_state["settings"].get("seat_start_dt")
                        seat_left_sec = compute_seat_left_seconds(
                            now, seat_start_dt, st.session_state.get("seat_extension_min", 0)
                        )

                        # 좌석 <= 59분이면: 다음 FOCUS 진입 차단 + 멈춤
                        if seat_left_sec is not None and (seat_left_sec <= SEAT_ALERT_WINDOW_SEC):
                            st.session_state["block_next_focus_until_seat_extended"] = True
                            st.session_state["seat_extension_context"] = "break"
                            st.session_state["extension_seat_left_sec"] = float(seat_left_sec)

                            st.session_state["show_extension_dialog"] = True
                            
                            # 현재 시각으로 종료 시각을 고정해 타이머 멈춤 (00:00)
                            st.session_state["phase_end_dt"] = now 
                            
                            st.rerun()
                            return

                    # 문제 없으면 정상적으로 FOCUS 시작
                    st.session_state["block_next_focus_until_seat_extended"] = False
                    _switch_phase(now, "FOCUS")
                    return

            # ---------------------------------------------------------
            # 5. (휴식 중) 좌석 59분 이하 알림 (1회성)
            # ---------------------------------------------------------
            if (not is_focus) and st.session_state["settings"].get("use_seat", False):
                if not is_seat_reset_window(now):
                    seat_start_dt = st.session_state["settings"].get("seat_start_dt")
                    seat_left_sec = compute_seat_left_seconds(
                        now, seat_start_dt, st.session_state.get("seat_extension_min", 0)
                    )

                    if seat_left_sec is not None and (seat_left_sec <= SEAT_ALERT_WINDOW_SEC):
                        if not st.session_state.get("seat_alert_shown_in_rest", False):
                            st.session_state["seat_alert_shown_in_rest"] = True
                            st.session_state["seat_extension_context"] = "break"
                            st.session_state["extension_seat_left_sec"] = float(seat_left_sec)

                            st.session_state["show_extension_dialog"] = True
                            st.rerun()
                            return
        run_timer_fragment()

        # 다이얼로그는 여기서 "딱 하나"만 오픈 (fragment 밖)
        if st.session_state.get("show_seat_settings", False):
            seat_settings_dialog()
        elif st.session_state.get("show_start_setup", False):
            start_setup_dialog()
        elif st.session_state.get("show_extension_dialog", False):
            extension_dialog()
        elif st.session_state.get("show_stop_dialog", False):
            stop_dialog()
        elif st.session_state.get("show_pause_dialog", False):
            pause_dialog()

        # fragment에서 다이얼로그 띄우라고 플래그만 켠 경우, 메인 rerun으로 반영
        if st.session_state.get("need_main_rerun", False):
            st.session_state["need_main_rerun"] = False
            st.rerun()

        st.write("")
        cA, cB = st.columns(2, gap="small")
        if st.session_state.get("running", False):
            with cA:
                if st.button("⏸️ 중단", width="stretch", disabled=st.session_state.get("paused", False)):
                    _open_dialog("show_pause_dialog")
            with cB:
                if st.button("🏁 종료", width="stretch"):
                    _open_dialog("show_stop_dialog")

    with col_todo:
        st.markdown(
            '<div class="timer-title" style="text-align:left;">📝 To-Do List</div>',
            unsafe_allow_html=True,
        )

        with st.form(key="todo_form", clear_on_submit=True):
            f_col1, f_col2 = st.columns([4, 1])
            new_task = f_col1.text_input("할 일", label_visibility="collapsed", placeholder="할 일 추가")
            submit = f_col2.form_submit_button("추가", width="stretch")

        # TODO 추가: 새 항목이 "맨 아래"에 보이도록
        if submit and new_task:
            max_order = pd.read_sql(
                "SELECT MAX(task_order) AS m FROM todos WHERE status != 'deleted'",
                conn
            ).iloc[0, 0]
            if pd.isna(max_order):
                max_order = 0

            conn.cursor().execute(
                "INSERT INTO todos (task, status, date, is_subtask, task_order) VALUES (?, ?, ?, ?, ?)",
                (new_task, "pending", datetime.now().strftime("%Y-%m-%d"), 0, int(max_order) + 1),
            )
            conn.commit()
            st.rerun()

        df_todos = pd.read_sql("SELECT * FROM todos WHERE status != 'deleted' ORDER BY task_order ASC", conn)
        if not df_todos.empty:
            for _, row in df_todos.iterrows():
                c_chk, c_txt, c_del = st.columns([0.6, 8, 1.2])
                is_done = row["status"] == "done"

                with c_chk:
                    if st.checkbox(
                        f"완료_{row['id']}",
                        value=is_done,
                        key=f"chk_{row['id']}",
                        label_visibility="collapsed",
                    ) != is_done:
                        new_status = "pending" if is_done else "done"
                        conn.cursor().execute("UPDATE todos SET status=? WHERE id=?", (new_status, int(row["id"])))
                        conn.commit()
                        st.rerun()

                with c_txt:
                    if is_done:
                        st.markdown(
                            f"<div class='todo-text'><span class='todo-done'>{row['task']}</span></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        new_val = st.text_input(
                            "수정", value=str(row["task"]), key=f"edit_{row['id']}", label_visibility="collapsed"
                        )
                        if new_val != row["task"]:
                            conn.cursor().execute("UPDATE todos SET task=? WHERE id=?", (new_val, int(row["id"])))
                            conn.commit()
                            st.rerun()

                with c_del:
                    if st.button("삭제", key=f"del_{row['id']}", width="stretch"):
                        conn.cursor().execute("UPDATE todos SET status='deleted' WHERE id=?", (int(row["id"]),))
                        conn.commit()
                        st.rerun()
        else:
            st.info("할 일이 없습니다.")


with tab2:
    st.header("📊 학습 분석 리포트")

    df_s = pd.read_sql("SELECT * FROM study_sessions", conn)
    df_i = pd.read_sql("SELECT * FROM interruptions", conn)

    if not df_s.empty:
        df_s["start_time"] = pd.to_datetime(df_s["start_time"], errors="coerce")
        df_s["focus_minutes"] = pd.to_numeric(df_s["focus_minutes"], errors="coerce").fillna(0)

        period = st.radio("조회 기간 선택", ["최근 1주일", "최근 1개월"], horizontal=True)
        days = 7 if period == "최근 1주일" else 30
        cutoff = datetime.now() - timedelta(days=days)
        df_filtered = df_s[df_s["start_time"] >= cutoff].copy()

        st.subheader(f"📈 {period} 집중 시간 추이")
        df_daily = df_filtered.groupby(df_filtered["start_time"].dt.date)["focus_minutes"].sum().reset_index()
        df_daily.columns = ["날짜", "집중시간(분)"]
        df_daily = df_daily.sort_values("날짜")

        if not df_daily.empty:
            df_daily["날짜_dt"] = pd.to_datetime(df_daily["날짜"])
            fig = px.line(df_daily, x="날짜_dt", y="집중시간(분)", markers=True, text="집중시간(분)")
            fig.update_layout(hovermode="x unified", xaxis_title=None)
            fig.update_traces(line_width=3, marker_size=10, textposition="top center")
            st.plotly_chart(fig, width="stretch")

        st.write("")
        st.subheader("🔍 심층 분석")
        row1_c1, row1_c2 = st.columns([1.5, 1])

        with row1_c1:
            st.markdown("**📅 집중 리듬 (요일 x 시간대)**")
            if not df_filtered.empty:
                df_hm = df_filtered.copy()
                df_hm["weekday"] = df_hm["start_time"].dt.day_name()
                df_hm["hour"] = df_hm["start_time"].dt.hour

                days_order = list(calendar.day_name)
                df_hm["weekday"] = pd.Categorical(df_hm["weekday"], categories=days_order, ordered=True)

                heatmap_data = df_hm.groupby(["weekday", "hour"], observed=False)["focus_minutes"].sum().reset_index()
                pivot_table = heatmap_data.pivot(index="hour", columns="weekday", values="focus_minutes").fillna(0)

                all_hours = list(range(24))
                pivot_table = pivot_table.reindex(index=all_hours, columns=days_order, fill_value=0)

                fig_hm = px.imshow(
                    pivot_table,
                    labels=dict(x="요일", y="시간", color="분"),
                    x=days_order,
                    y=all_hours,
                    color_continuous_scale="Reds",
                    aspect="auto",
                )

                tickvals = [0, 6, 12, 18, 23]
                ticktext = ["00:00", "06:00", "12:00", "18:00", "24:00"]

                fig_hm.update_yaxes(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    autorange="reversed",
                )
                fig_hm.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
                st.plotly_chart(fig_hm, width="stretch")
            else:
                st.info("데이터가 부족합니다.")

        with row1_c2:
            st.markdown("**🛑 방해 요인 비율 (집중 시간 기준)**")
            if not df_i.empty:
                df_i_focus = df_i.copy()
                if "phase" in df_i_focus.columns:
                    df_i_focus = df_i_focus[df_i_focus["phase"] == "FOCUS"].copy()
                else:
                    df_i_focus = df_i_focus.iloc[0:0].copy()

                rest_cnt = 0
                if "phase" in df_i.columns:
                    rest_cnt = int((df_i["phase"] == "REST").sum())
                if rest_cnt > 0:
                    st.caption(f"※ 참고: 휴식(REST) 중 기록 {rest_cnt}건은 방해요인 집계에서 제외됨")

                if not df_i_focus.empty:
                    reason_counts = df_i_focus["reason"].value_counts().reset_index()
                    reason_counts.columns = ["reason", "count"]

                    fig_pie = px.pie(reason_counts, values="count", names="reason", hole=0.4)
                    fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, showlegend=False)
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig_pie, width="stretch")
                else:
                    st.info("집중(FOCUS) 중 중단/종료 기록이 없습니다.")
            else:
                st.info("중단 기록이 없습니다.")

    else:
        st.info("📊 아직 학습 기록이 없습니다. 타이머를 사용해 첫 데이터를 만들어보세요!")

    st.divider()
    st.subheader("✨ AI 상세 리포트")
    if st.button("✨ 상세 분석 리포트 생성", width="stretch"):
        if not api_key:
            st.error("사이드바에 Gemini API Key를 입력해주세요.")
        else:
            with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                period = "최근 1주일"
                days = 7
                report = ai_generate_report(api_key, df_s, df_i, period, days)
                st.success("분석 완료!")
                st.markdown(report)

