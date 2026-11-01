import json
import joblib
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from cricket_qa import (
    answer_historical_question,
    format_historical_answer,
)
# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="BOUNDARY | Cricket Intelligence & Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
CALIBRATED_MODEL_PATH = BASE_DIR / "cricket_win_model_calibrated.joblib"
BASE_MODEL_PATH = BASE_DIR / "cricket_win_model.joblib"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "qwen2.5:3b"

# Popular Teams with Franchise Colors
TEAMS_DB = {
    "Chennai Super Kings": {"short": "CSK", "color": "#FDB913", "text_dark": True},
    "Mumbai Indians": {"short": "MI", "color": "#004BA0", "text_dark": False},
    "Royal Challengers Bengaluru": {"short": "RCB", "color": "#DC2626", "text_dark": False},
    "Kolkata Knight Riders": {"short": "KKR", "color": "#7C3AED", "text_dark": False},
    "Sunrisers Hyderabad": {"short": "SRH", "color": "#F97316", "text_dark": False},
    "Rajasthan Royals": {"short": "RR", "color": "#EC4899", "text_dark": False},
    "Gujarat Titans": {"short": "GT", "color": "#1E3A8A", "text_dark": False},
    "Delhi Capitals": {"short": "DC", "color": "#0284C7", "text_dark": False},
    "Lucknow Super Giants": {"short": "LSG", "color": "#06B6D4", "text_dark": False},
    "Punjab Kings": {"short": "PBKS", "color": "#E11D48", "text_dark": False},
    "India": {"short": "IND", "color": "#2563EB", "text_dark": False},
    "Australia": {"short": "AUS", "color": "#FACC15", "text_dark": True},
    "England": {"short": "ENG", "color": "#E11D48", "text_dark": False},
    "South Africa": {"short": "SA", "color": "#16A34A", "text_dark": False},
    "Pakistan": {"short": "PAK", "color": "#15803D", "text_dark": False},
    "New Zealand": {"short": "NZ", "color": "#334155", "text_dark": False},
    "West Indies": {"short": "WI", "color": "#991B1B", "text_dark": False},
    "Afghanistan": {"short": "AFG", "color": "#0284C7", "text_dark": False},
}

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
default_states = {
    "nav_tab": "Match Center",
    "score": 120,
    "wickets": 3,
    "target": 180,
    "overs": 15,
    "balls": 0,
    "team_a": "Chennai Super Kings",
    "team_b": "Mumbai Indians",
    "ball_history": ["1", "4", "0", "2", "6", "1"],
    "action_stack": [],
    "chat_history": [],
    "last_match_context": None,
    "match_result": None,
    "custom_team_a": "",
    "custom_team_b": "",
    "show_ai_chat": False,
}

for k, v in default_states.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sync widget state keys if not already present
for key in ["score", "wickets", "target", "overs", "balls"]:
    w_key = f"{key}_input"
    if w_key not in st.session_state:
        st.session_state[w_key] = st.session_state[key]


# =========================================================
# ULTRA-PREMIUM SPORTS BROADCAST DESIGN SYSTEM (CSS)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #070b14;
    --card-bg: rgba(15, 23, 42, 0.78);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-hover: rgba(30, 41, 59, 0.9);
    --lime-neon: #a3e635;
    --emerald-neon: #10b981;
    --cyan-neon: #38bdf8;
    --rose-neon: #f43f5e;
    --amber-neon: #f59e0b;
    --text-white: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
}

/* Background Atmosphere */
.stApp {
    background: 
        radial-gradient(circle at 10% 12%, rgba(16, 185, 129, 0.07) 0%, transparent 40%),
        radial-gradient(circle at 88% 80%, rgba(56, 189, 248, 0.06) 0%, transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(163, 230, 53, 0.02) 0%, transparent 60%),
        #070b14 !important;
    color: var(--text-white);
    font-family: 'Plus Jakarta Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}

.block-container {
    max-width: 1420px;
    padding-top: 5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 3.5rem !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: var(--text-white) !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.5px;
    font-weight: 700 !important;
}

/* Brand Navbar */
.brand-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(18, 30, 49, 0.9));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    margin-bottom: 18px;
    box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.brand-icon-box {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(56, 189, 248, 0.25));
    border: 1px solid rgba(163, 230, 53, 0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 0 18px rgba(16, 185, 129, 0.25);
}

.brand-title {
    font-family: 'Outfit', sans-serif;
    font-size: 25px;
    font-weight: 900;
    letter-spacing: 0.5px;
    color: #ffffff;
    line-height: 1.15;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
}

.brand-title .brand-accent {
    color: #10b981;
    text-shadow: 0 0 16px rgba(16, 185, 129, 0.4);
}

.brand-sub {
    font-size: 11.5px;
    letter-spacing: 1.6px;
    font-weight: 700;
    color: #a3e635;
    text-transform: uppercase;
    margin-top: 3px;
}

.live-pulse {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #ef4444;
    border-radius: 50%;
    box-shadow: 0 0 8px #ef4444;
    animation: pulse 1.6s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.85; }
    50% { transform: scale(1.25); opacity: 1; box-shadow: 0 0 12px #ef4444; }
    100% { transform: scale(0.95); opacity: 0.85; }
}

/* Glass Cards */
.glass-panel {
    background: var(--card-bg);
    backdrop-filter: blur(14px);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.45);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.glass-panel:hover {
    border-color: rgba(255, 255, 255, 0.14);
}

/* Scoreboard Broadcast Banner */
.scoreboard-hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(13, 27, 42, 0.85) 50%, rgba(16, 38, 30, 0.9) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 24px;
    padding: 26px 30px;
    margin-bottom: 22px;
    box-shadow: 0 14px 40px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.scoreboard-hero::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 300px;
    height: 100%;
    background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 70%);
    pointer-events: none;
}

.scoreboard-teams {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
    flex-wrap: wrap;
    gap: 14px;
}

.team-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 16px;
    border-radius: 14px;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: -0.2px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.match-score-display {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(38px, 5.5vw, 64px);
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -1.5px;
}

.match-score-overs {
    font-size: clamp(18px, 2.5vw, 26px);
    color: var(--text-muted);
    font-weight: 500;
    margin-left: 12px;
}

.metric-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* Win Probability Big Card */
.win-prob-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(18, 30, 49, 0.9));
    border: 1px solid rgba(163, 230, 53, 0.28);
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 30px -5px rgba(16, 185, 129, 0.15);
}

.win-prob-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(48px, 6vw, 68px);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -1.5px;
    margin: 10px 0 6px;
}

/* Custom Metric Tiles */
.stat-tile {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 18px;
    transition: all 0.2s ease;
}

.stat-tile:hover {
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-2px);
}

.stat-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.stat-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
}

/* Ball-by-ball pill indicators */
.ball-strip {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
    margin: 12px 0;
}

.ball-item {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 13px;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.35);
}

.ball-dot { background: #1e293b; color: #94a3b8; border: 1px solid #334155; }
.ball-run { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
.ball-four { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #10b981; font-weight: 800; }
.ball-six { background: linear-gradient(135deg, #10b981, #a3e635); color: #06240c; font-weight: 900; }
.ball-wicket { background: #ef4444; color: #ffffff; font-weight: 900; box-shadow: 0 0 10px rgba(239, 68, 68, 0.6); }

/* Form Inputs Styling */
.stTextInput input, .stNumberInput input {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25) !important;
}

div[data-baseweb="select"] > div {
    background: #0f172a !important;
    border-color: #334155 !important;
    border-radius: 12px !important;
}

/* Button Upgrades */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px;
    padding: 8px 18px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #f8fafc !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover {
    border-color: #10b981 !important;
    color: #10b981 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px -5px rgba(16, 185, 129, 0.25) !important;
}

/* Primary Action Buttons */
.primary-btn .stButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(16, 185, 129, 0.3) !important;
}

.primary-btn .stButton > button:hover {
    background: linear-gradient(135deg, #059669, #047857) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.45) !important;
}

/* Action ball buttons */
.ball-btn-dot .stButton > button { background: #1e293b !important; color: #94a3b8 !important; }
.ball-btn-run .stButton > button { background: #0c4a6e !important; color: #38bdf8 !important; border-color: #0284c7 !important; }
.ball-btn-four .stButton > button { background: #064e3b !important; color: #34d399 !important; border-color: #059669 !important; }
.ball-btn-six .stButton > button { background: linear-gradient(135deg, #15803d, #65a30d) !important; color: #ffffff !important; font-weight: 800 !important; }
.ball-btn-wkt .stButton > button { background: #991b1b !important; color: #fecaca !important; border-color: #dc2626 !important; }

/* Chat Custom Styling */
div[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important;
}

[data-testid="stChatInput"] {
    border-color: #334155 !important;
}

/* Streamlit Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 8px 20px;
    font-weight: 700;
    font-size: 14px;
    color: var(--text-muted);
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #10b981 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* Segmented Control Pill Bar */
div[data-testid="stSegmentedControl"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid="stSegmentedControl"] button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    color: var(--text-muted) !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(56, 189, 248, 0.15)) !important;
    color: #a3e635 !important;
    border: 1px solid rgba(163, 230, 53, 0.4) !important;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.25) !important;
}

/* Streamlit metric overrides */
div[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADER
# =========================================================
@st.cache_resource
def load_prediction_model():
    """Loads calibrated model if available, fallback to standard model."""
    if CALIBRATED_MODEL_PATH.exists():
        saved = joblib.load(CALIBRATED_MODEL_PATH)
        return saved["model"], saved["features"], True
    elif BASE_MODEL_PATH.exists():
        saved = joblib.load(BASE_MODEL_PATH)
        return saved["model"], saved["features"], False
    else:
        return None, None, False


model, features, is_calibrated = load_prediction_model()

if model is None:
    st.error("⚠️ Model file not found. Ensure `cricket_win_model.joblib` is present in the application folder.")
    st.stop()


# =========================================================
# MATCH CALCULATION ENGINE
# =========================================================
def calculate_match_state(score, wickets, legal_balls, target):
    """Computes cricket equation, required rates, and model win probability."""
    balls_remaining = max(0, 120 - legal_balls)
    runs_remaining = max(0, target - score)
    overs_completed = legal_balls / 6.0

    current_rr = score / overs_completed if overs_completed > 0 else 0.0

    if score >= target:
        status = "won"
        probability = 100.0
        required_rr = 0.0
    elif wickets >= 10 or (legal_balls >= 120 and score < target):
        status = "lost"
        probability = 0.0
        required_rr = 99.9
    elif legal_balls == 120 and score == target - 1:
        status = "tied"
        probability = 50.0
        required_rr = 0.0
    else:
        status = "active"
        required_rr = (runs_remaining / (balls_remaining / 6.0)) if balls_remaining > 0 else 99.9

        input_df = pd.DataFrame([{
            "score": score,
            "wickets_lost": wickets,
            "legal_balls": legal_balls,
            "balls_remaining": balls_remaining,
            "target": target,
            "runs_remaining": runs_remaining,
            "current_run_rate": current_rr,
            "required_run_rate": required_rr,
        }])[features]

        raw_prob = float(model.predict_proba(input_df)[0][1] * 100)
        # Clamped for realistic active in-game boundaries
        probability = min(99.4, max(0.6, raw_prob))

    # Phase detection
    if legal_balls <= 36:
        phase = "Powerplay (Overs 1-6)"
        phase_badge = "🚀 POWERPLAY"
    elif legal_balls <= 90:
        phase = "Middle Overs (Overs 7-15)"
        phase_badge = "⚙️ MIDDLE OVERS"
    elif legal_balls <= 114:
        phase = "Death Overs (Overs 16-19)"
        phase_badge = "🔥 DEATH OVERS"
    else:
        phase = "Final Over (Over 20)"
        phase_badge = "⚡ CLUTCH FINAL OVER"

    # Momentum Assessment
    if probability >= 75:
        momentum_text = "Dominating the Chase"
        momentum_color = "#10b981"
    elif probability >= 55:
        momentum_text = "Chasing Team In Advantage"
        momentum_color = "#34d399"
    elif probability >= 45:
        momentum_text = "Even Contest (50-50 Thriller)"
        momentum_color = "#f59e0b"
    elif probability >= 25:
        momentum_text = "Bowling Team Pressuring"
        momentum_color = "#fb923c"
    else:
        momentum_text = "Bowling Team In Command"
        momentum_color = "#f43f5e"

    runs_per_ball = (runs_remaining / balls_remaining) if balls_remaining > 0 else 0.0
    boundaries_needed = max(0, int(runs_remaining / 4.5))

    return {
        "score": score,
        "wickets": wickets,
        "legal_balls": legal_balls,
        "overs_str": f"{legal_balls // 6}.{legal_balls % 6}",
        "target": target,
        "balls_remaining": balls_remaining,
        "runs_remaining": runs_remaining,
        "current_rr": current_rr,
        "required_rr": required_rr,
        "probability": probability,
        "bowling_prob": 100.0 - probability,
        "status": status,
        "phase": phase,
        "phase_badge": phase_badge,
        "momentum_text": momentum_text,
        "momentum_color": momentum_color,
        "runs_per_ball": runs_per_ball,
        "boundaries_needed": boundaries_needed,
    }


# =========================================================
# ADVANCED VISUALIZATIONS
# =========================================================
def build_dual_win_bar(prob, team_a, team_b, team_a_color, team_b_color):
    """Draws a sleek split win-probability bar with team colors."""
    fig = go.Figure()
    
    # Team A bar
    fig.add_trace(go.Bar(
        x=[prob],
        y=["Probability"],
        orientation="h",
        name=team_a,
        marker=dict(
            color=team_a_color,
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=f"{team_a}: {prob:.1f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=14, family="Space Grotesk"),
        hovertemplate=f"<b>{team_a}</b>: {prob:.1f}%<extra></extra>",
    ))
    
    # Team B bar
    fig.add_trace(go.Bar(
        x=[100 - prob],
        y=["Probability"],
        orientation="h",
        name=team_b,
        marker=dict(
            color=team_b_color,
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        ),
        text=f"{team_b}: {100 - prob:.1f}%",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=14, family="Space Grotesk"),
        hovertemplate=f"<b>{team_b}</b>: {100 - prob:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        height=68,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def build_gauge_dial(probability, batting_team):
    """High-tech semi-circular radial probability meter."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': "%",
            'font': {'size': 44, 'color': '#ffffff', 'family': 'Space Grotesk'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#475569',
                'tickfont': {'color': '#94a3b8', 'size': 10}
            },
            'bar': {'color': '#10b981', 'thickness': 0.28},
            'bgcolor': "rgba(15, 23, 42, 0.7)",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.22)'},
                {'range': [25, 45], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [45, 55], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [55, 75], 'color': 'rgba(56, 189, 248, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.25)'},
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.85,
                'value': probability
            }
        }
    ))
    fig.update_layout(
        height=195,
        margin=dict(l=15, r=15, t=10, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Space Grotesk")
    )
    return fig


def build_rate_pressure_chart(crr, rrr):
    """Visualizes run rate gap and chase pressure."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Current Run Rate", "Required Run Rate"],
        y=[crr, rrr],
        marker=dict(
            color=["#38bdf8", "#f43f5e" if rrr > crr + 2 else ("#f59e0b" if rrr > crr else "#10b981")],
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        ),
        text=[f"{crr:.2f}", f"{rrr:.2f}"],
        textposition="outside",
        textfont=dict(color="#ffffff", size=14, family="Space Grotesk")
    ))

    max_y = max(14.0, max(crr, rrr) * 1.25)
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        yaxis=dict(range=[0, max_y], gridcolor="rgba(255,255,255,0.06)", title="Runs / Over"),
        xaxis=dict(tickfont=dict(color="#f8fafc", size=12)),
        showlegend=False,
    )
    return fig


# =========================================================
# HELPER: QUICK PRESETS & BALL ACTIONS
# =========================================================
def apply_preset(name):
    presets = {
        "thriller": {
            "score": 170, "wickets": 6, "target": 182, "overs": 19, "balls": 0,
            "team_a": "Chennai Super Kings", "team_b": "Mumbai Indians"
        },
        "death_overs": {
            "score": 142, "wickets": 4, "target": 188, "overs": 16, "balls": 0,
            "team_a": "Royal Challengers Bengaluru", "team_b": "Kolkata Knight Riders"
        },
        "middle_squeeze": {
            "score": 96, "wickets": 3, "target": 174, "overs": 12, "balls": 0,
            "team_a": "India", "team_b": "Australia"
        },
        "powerplay_blitz": {
            "score": 68, "wickets": 1, "target": 205, "overs": 6, "balls": 0,
            "team_a": "Sunrisers Hyderabad", "team_b": "Rajasthan Royals"
        },
        "collapse": {
            "score": 46, "wickets": 5, "target": 168, "overs": 8, "balls": 0,
            "team_a": "Pakistan", "team_b": "England"
        }
    }
    if name in presets:
        p = presets[name]
        st.session_state.score = p["score"]
        st.session_state.wickets = p["wickets"]
        st.session_state.target = p["target"]
        st.session_state.overs = p["overs"]
        st.session_state.balls = p["balls"]
        st.session_state.team_a = p["team_a"]
        st.session_state.team_b = p["team_b"]
        st.session_state.ball_history = []
        st.session_state.action_stack = []
        
        # update inputs
        for key in ["score", "wickets", "target", "overs", "balls"]:
            st.session_state[f"{key}_input"] = p[key]


def register_ball(ball_type):
    """Executes a ball event and calculates the impact dynamically."""
    current_legal = st.session_state.overs * 6 + st.session_state.balls
    if current_legal >= 120 or st.session_state.wickets >= 10 or st.session_state.score >= st.session_state.target:
        return

    # Save state for undo
    st.session_state.action_stack.append({
        "score": st.session_state.score,
        "wickets": st.session_state.wickets,
        "overs": st.session_state.overs,
        "balls": st.session_state.balls,
        "history": list(st.session_state.ball_history)
    })

    if ball_type == "0":
        st.session_state.balls += 1
        st.session_state.ball_history.append("0")
    elif ball_type in ["1", "2", "3"]:
        runs = int(ball_type)
        st.session_state.score += runs
        st.session_state.balls += 1
        st.session_state.ball_history.append(str(runs))
    elif ball_type == "4":
        st.session_state.score += 4
        st.session_state.balls += 1
        st.session_state.ball_history.append("4")
    elif ball_type == "6":
        st.session_state.score += 6
        st.session_state.balls += 1
        st.session_state.ball_history.append("6")
    elif ball_type == "W":
        st.session_state.wickets = min(10, st.session_state.wickets + 1)
        st.session_state.balls += 1
        st.session_state.ball_history.append("W")
    elif ball_type == "Wd":
        st.session_state.score += 1
        st.session_state.ball_history.append("Wd")
    elif ball_type == "Nb":
        st.session_state.score += 1
        st.session_state.ball_history.append("Nb")

    # Handle over completion
    if st.session_state.balls >= 6:
        st.session_state.overs += 1
        st.session_state.balls = 0

    # Sync input widgets
    for k in ["score", "wickets", "overs", "balls"]:
        st.session_state[f"{k}_input"] = st.session_state[k]


def undo_ball():
    if st.session_state.action_stack:
        last = st.session_state.action_stack.pop()
        st.session_state.score = last["score"]
        st.session_state.wickets = last["wickets"]
        st.session_state.overs = last["overs"]
        st.session_state.balls = last["balls"]
        st.session_state.ball_history = last["history"]
        for k in ["score", "wickets", "overs", "balls"]:
            st.session_state[f"{k}_input"] = st.session_state[k]


# =========================================================
# HEADER & BRAND BAR
# =========================================================
@st.cache_data(ttl=8)
def check_ollama_status():
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False

ollama_online = check_ollama_status()

brand_col, chat_btn_col, info_col = st.columns([2.2, 0.95, 0.95])

with brand_col:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-left">
            <div class="brand-icon-box">🏏</div>
            <div>
                <div class="brand-title">BOUNDARY <span class="brand-accent">INTELLIGENCE</span></div>
                <div class="brand-sub">T20 Live Win Probability & Tactical Simulator</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="live-pulse"><span class="pulse-dot"></span> LIVE IN-PLAY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with chat_btn_col:
    chat_open = st.session_state.show_ai_chat
    btn_text = "✕ Close AI Box" if chat_open else "🤖 AI Strategist 💬"
    btn_type = "primary" if chat_open else "secondary"
    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(15,23,42,0.95), rgba(18,30,49,0.9)); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:12px 16px; margin-bottom:18px; box-shadow:0 12px 35px -8px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.12);">
        <div style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:6px; letter-spacing:0.8px;">
            Copilot Panel
        </div>
    """, unsafe_allow_html=True)
    if st.button(btn_text, key="top_toggle_ai_drawer", use_container_width=True, type=btn_type):
        st.session_state.show_ai_chat = not st.session_state.show_ai_chat
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    calib_badge = '<span style="color:#10b981;font-weight:700">● CALIBRATED AI MODEL</span>' if is_calibrated else '<span style="color:#38bdf8;font-weight:700">● STANDARD RF MODEL</span>'
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, rgba(15,23,42,0.95), rgba(18,30,49,0.9)); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:15px 20px; margin-bottom:18px; font-size:12px; line-height:1.5; box-shadow:0 12px 35px -8px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.12);">
        <div style="color:#94a3b8; font-weight:700; letter-spacing:0.8px; text-transform:uppercase;">Model Status</div>
        <div>{calib_badge} (Brier optimized)</div>
        <div style="color:#64748b; font-size:11px;">Features: 8 In-game Match Vectors</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TOP NAVIGATION (PILL BAR)
# =========================================================
tabs = ["🏟️ Match Center", "⚡ Live Ball Simulator", "🔮 What-If Scenario Matrix", "🤖 AI Cricket Strategist"]
selected_tab = st.segmented_control(
    "Navigation",
    tabs,
    default=st.session_state.nav_tab if st.session_state.nav_tab in tabs else tabs[0],
    label_visibility="collapsed"
)
if selected_tab:
    st.session_state.nav_tab = selected_tab
    if selected_tab == "🤖 AI Cricket Strategist":
        st.session_state.show_ai_chat = True

active_tab = st.session_state.nav_tab

st.write("")


# =========================================================
# QUICK PRESETS BAR (Instantly load high-drama matches)
# =========================================================
with st.container():
    st.markdown("<div style='font-size:12px; font-weight:700; color:#94a3b8; letter-spacing:1px; margin-bottom:8px; text-transform:uppercase;'>⚡ Quick Load Match Scenarios</div>", unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6 = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    with p1:
        if st.button("🔥 Last Over Drama", use_container_width=True):
            apply_preset("thriller")
            st.rerun()
    with p2:
        if st.button("⚡ Death Tension", use_container_width=True):
            apply_preset("death_overs")
            st.rerun()
    with p3:
        if st.button("🎯 Middle Squeeze", use_container_width=True):
            apply_preset("middle_squeeze")
            st.rerun()
    with p4:
        if st.button("🚀 Powerplay Blitz", use_container_width=True):
            apply_preset("powerplay_blitz")
            st.rerun()
    with p5:
        if st.button("🌪 Top Collapse", use_container_width=True):
            apply_preset("collapse")
            st.rerun()
    with p6:
        with st.popover("⚙️ Teams", use_container_width=True):
            st.markdown("#### Team Setup")
            all_teams = list(TEAMS_DB.keys())
            t_a = st.selectbox("Chasing Team (Batting)", all_teams, index=all_teams.index(st.session_state.team_a) if st.session_state.team_a in all_teams else 0)
            t_b = st.selectbox("Bowling Team", all_teams, index=all_teams.index(st.session_state.team_b) if st.session_state.team_b in all_teams else 1)
            st.session_state.team_a = t_a
            st.session_state.team_b = t_b


# Retrieve team details & colors
team_a_info = TEAMS_DB.get(st.session_state.team_a, {"short": "BAT", "color": "#10b981"})
team_b_info = TEAMS_DB.get(st.session_state.team_b, {"short": "BOWL", "color": "#0284c7"})

# Synchronize current numerical match state
legal_balls_calc = st.session_state.overs * 6 + st.session_state.balls
match = calculate_match_state(
    st.session_state.score,
    st.session_state.wickets,
    legal_balls_calc,
    st.session_state.target
)
st.session_state.match_result = match

# Context for AI
st.session_state.last_match_context = f"""
MATCH CONTEXT:
Chasing Team: {st.session_state.team_a}
Bowling Team: {st.session_state.team_b}
Score: {match['score']}/{match['wickets']} in {match['overs_str']} overs
Target: {match['target']}
Equation: {match['runs_remaining']} runs required from {match['balls_remaining']} balls
Current Run Rate (CRR): {match['current_rr']:.2f}
Required Run Rate (RRR): {match['required_rr']:.2f}
Match Phase: {match['phase']}
Win Probability for Chasing Team: {match['probability']:.1f}%
Status: {match['status']}
"""


# =========================================================
# BROADCAST SCOREBOARD HERO BANNER (Visible across views)
# =========================================================
rr_gap = match['required_rr'] - match['current_rr']
gap_badge_color = "#f43f5e" if rr_gap > 2 else ("#f59e0b" if rr_gap > 0 else "#10b981")
gap_text = f"+{rr_gap:.2f} RRR Pressure" if rr_gap > 0 else f"{abs(rr_gap):.2f} Ahead of RRR"

st.markdown(f"""
<div class="scoreboard-hero">
    <div class="scoreboard-teams">
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="team-pill" style="background:{team_a_info['color']}; color:#ffffff;">
                🏏 {st.session_state.team_a}
            </div>
            <span style="font-weight:700; color:#64748b; font-size:13px;">VS</span>
            <div class="team-pill" style="background:{team_b_info['color']}; color:#ffffff;">
                🛡️ {st.session_state.team_b}
            </div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="metric-pill" style="background:rgba(255,255,255,0.08); color:#f8fafc;">{match['phase_badge']}</span>
            <span class="metric-pill" style="background:{gap_badge_color}22; color:{gap_badge_color}; border:1px solid {gap_badge_color}55;">{gap_text}</span>
            <span class="metric-pill" style="background:{match['momentum_color']}22; color:{match['momentum_color']}; border:1px solid {match['momentum_color']}55;">{match['momentum_text']}</span>
        </div>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:16px;">
        <div>
            <span class="match-score-display">{match['score']}<span style="color:#ef4444">/{match['wickets']}</span></span>
            <span class="match-score-overs">({match['overs_str']} ov)</span>
        </div>
        <div style="font-family:'Space Grotesk', sans-serif; font-size: clamp(18px, 2.5vw, 24px); font-weight:700; color:#38bdf8;">
            Target: {match['target']} <span style="color:#64748b;">·</span> Need <span style="color:#a3e635;">{match['runs_remaining']}</span> off <span style="color:#a3e635;">{match['balls_remaining']}</span> balls
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODULAR VIEW RENDERERS
# =========================================================
def render_match_center_content(match, team_a_info, team_b_info):
    """Renders the primary match center analytics and controls."""
    if st.session_state.show_ai_chat:
        col_dash, col_controls = st.columns([1.2, 1.0], gap="medium")
    else:
        col_dash, col_controls = st.columns([1.5, 1.0], gap="large")

    with col_dash:
        # Dual-Team Probability Split
        st.markdown("<div style='font-size:12px; font-weight:700; color:#94a3b8; letter-spacing:1px; margin-bottom:8px; text-transform:uppercase;'>Head-to-Head Win Probability</div>", unsafe_allow_html=True)
        st.plotly_chart(
            build_dual_win_bar(match['probability'], st.session_state.team_a, st.session_state.team_b, team_a_info['color'], team_b_info['color']),
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.write("")
        # Dial + Run Rate Pressure Side-by-Side
        m_left, m_right = st.columns(2)
        with m_left:
            with st.container(border=True):
                st.markdown("<div style='font-size:12px; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Chasing Win Probability</div>", unsafe_allow_html=True)
                st.plotly_chart(
                    build_gauge_dial(match['probability'], st.session_state.team_a),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
                st.caption(f"Estimated by Random Forest classifier for {st.session_state.team_a}")

        with m_right:
            with st.container(border=True):
                st.markdown("<div style='font-size:12px; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Run Rate Battle (CRR vs RRR)</div>", unsafe_allow_html=True)
                st.plotly_chart(
                    build_rate_pressure_chart(match['current_rr'], match['required_rr']),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
                st.caption(f"Target pace: {match['runs_per_ball']:.2f} runs per ball remaining")

        # Chase Equation breakdown grid
        st.write("")
        st.markdown("<div style='font-size:12px; font-weight:700; color:#94a3b8; letter-spacing:1px; margin-bottom:12px; text-transform:uppercase;'>Live Chase Telemetry</div>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">Runs Needed</div>
                <div class="stat-number" style="color:#a3e635">{match['runs_remaining']}</div>
            </div>
            """, unsafe_allow_html=True)
        with g2:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">Balls Left</div>
                <div class="stat-number">{match['balls_remaining']}</div>
            </div>
            """, unsafe_allow_html=True)
        with g3:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">Req Run Rate</div>
                <div class="stat-number" style="color:{gap_badge_color}">{match['required_rr']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with g4:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">Est Boundaries</div>
                <div class="stat-number">~{match['boundaries_needed']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_controls:
        with st.container(border=True):
            st.markdown("### 🎛️ Match State Editor")
            st.caption("Adjust match numbers manually or use the Ball Simulator.")

            s1, s2 = st.columns(2)
            with s1:
                new_score = st.number_input("Score", 0, 450, st.session_state.score, key="score_input")
            with s2:
                new_wickets = st.number_input("Wickets", 0, 10, st.session_state.wickets, key="wickets_input")

            new_target = st.number_input("Target", 1, 451, st.session_state.target, key="target_input")

            o1, o2 = st.columns(2)
            with o1:
                new_overs = st.number_input("Overs Completed", 0, 20, st.session_state.overs, key="overs_input")
            with o2:
                new_balls = st.number_input("Balls in Over", 0, 5 if new_overs < 20 else 0, st.session_state.balls, key="balls_input")

            st.write("")
            c_btn = st.button("⚡ Recalculate Live Odds", use_container_width=True, type="primary")

            if c_btn:
                st.session_state.score = int(new_score)
                st.session_state.wickets = int(new_wickets)
                st.session_state.target = int(new_target)
                st.session_state.overs = int(new_overs)
                st.session_state.balls = int(new_balls)
                st.rerun()

        # Recent Ball Ticker
        with st.container(border=True):
            st.markdown("<div style='font-size:12px; font-weight:700; color:#94a3b8; text-transform:uppercase;'>Recent Deliveries</div>", unsafe_allow_html=True)
            if st.session_state.ball_history:
                balls_html = ""
                for b in st.session_state.ball_history[-12:]:
                    cls = "ball-dot" if b == "0" else ("ball-wicket" if b == "W" else ("ball-six" if b == "6" else ("ball-four" if b == "4" else "ball-run")))
                    balls_html += f'<div class="ball-item {cls}">{b}</div>'
                st.markdown(f'<div class="ball-strip">{balls_html}</div>', unsafe_allow_html=True)
            else:
                st.caption("No deliveries in history yet. Use the Live Ball Simulator to play out the overs.")


def render_ball_simulator_content(match):
    """Renders the interactive ball-by-ball game player."""
    st.markdown("### ⚡ Live Interactive Ball-by-Ball Simulator")
    st.markdown("<div style='color:#94a3b8; font-size:14px; margin-bottom:18px;'>Simulate each delivery in real-time. Watch how every single run, boundary, or wicket immediately swings the win probability.</div>", unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1.3, 1.0], gap="large")

    with b_col1:
        with st.container(border=True):
            st.markdown("#### 🎯 Next Delivery Action")
            st.caption("Click any event to simulate the next legal or illegal delivery:")

            # Delivery Action Buttons
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown('<div class="ball-btn-dot">', unsafe_allow_html=True)
                if st.button("⚪ Dot (0)", use_container_width=True, key="sim_btn_0"):
                    register_ball("0")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with r2:
                st.markdown('<div class="ball-btn-run">', unsafe_allow_html=True)
                if st.button("1️⃣ Single", use_container_width=True, key="sim_btn_1"):
                    register_ball("1")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with r3:
                st.markdown('<div class="ball-btn-run">', unsafe_allow_html=True)
                if st.button("2️⃣ Two Runs", use_container_width=True, key="sim_btn_2"):
                    register_ball("2")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with r4:
                st.markdown('<div class="ball-btn-run">', unsafe_allow_html=True)
                if st.button("3️⃣ Three", use_container_width=True, key="sim_btn_3"):
                    register_ball("3")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            r5, r6, r7, r8 = st.columns(4)
            with r5:
                st.markdown('<div class="ball-btn-four">', unsafe_allow_html=True)
                if st.button("4️⃣ FOUR! (+4)", use_container_width=True, key="sim_btn_4"):
                    register_ball("4")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with r6:
                st.markdown('<div class="ball-btn-six">', unsafe_allow_html=True)
                if st.button("6️⃣ SIX! (+6)", use_container_width=True, key="sim_btn_6"):
                    register_ball("6")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with r7:
                st.markdown('<div class="ball-btn-wkt">', unsafe_allow_html=True)
                if st.button("⚡ WICKET!", use_container_width=True, key="sim_btn_w"):
                    register_ball("W")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with r8:
                if st.button("➕ Wide/NB", use_container_width=True, key="sim_btn_extra"):
                    register_ball("Wd")
                    st.rerun()

            st.write("")
            u1, u2 = st.columns(2)
            with u1:
                if st.button("↩️ Undo Last Ball", use_container_width=True, key="sim_btn_undo"):
                    undo_ball()
                    st.rerun()
            with u2:
                if st.button("🔄 Clear History", use_container_width=True, key="sim_btn_clear"):
                    st.session_state.ball_history = []
                    st.session_state.action_stack = []
                    st.rerun()

        # Ball Ticker
        st.markdown("#### 📜 Over Deliveries Strip")
        if st.session_state.ball_history:
            balls_html = ""
            for b in st.session_state.ball_history:
                cls = "ball-dot" if b == "0" else ("ball-wicket" if b == "W" else ("ball-six" if b == "6" else ("ball-four" if b == "4" else "ball-run")))
                balls_html += f'<div class="ball-item {cls}">{b}</div>'
            st.markdown(f'<div class="ball-strip">{balls_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No balls recorded in this session. Click any delivery button above to play out balls.")

    with b_col2:
        with st.container(border=True):
            st.markdown("#### 📊 Dynamic Win Impact")
            st.plotly_chart(
                build_gauge_dial(match['probability'], st.session_state.team_a),
                use_container_width=True,
                config={"displayModeBar": False}
            )
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:16px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#94a3b8; font-size:13px;">Chasing Probability:</span>
                    <strong style="color:#10b981; font-size:15px;">{match['probability']:.1f}%</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#94a3b8; font-size:13px;">Bowling Probability:</span>
                    <strong style="color:#38bdf8; font-size:15px;">{match['bowling_prob']:.1f}%</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#94a3b8; font-size:13px;">Runs Required:</span>
                    <strong style="color:#f8fafc; font-size:15px;">{match['runs_remaining']} off {match['balls_remaining']}b</strong>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#94a3b8; font-size:13px;">Current Phase:</span>
                    <strong style="color:#a3e635; font-size:13px;">{match['phase']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_what_if_matrix_content(match):
    """Renders the tactical sensitivity simulation matrix."""
    st.markdown("### 🔮 Tactical What-If Scenario Lab")
    st.markdown("<div style='color:#94a3b8; font-size:14px; margin-bottom:18px;'>Simulate how the win probability shifts across different outcomes in the immediate next over (6 balls).</div>", unsafe_allow_html=True)

    curr_balls = match['legal_balls']
    curr_score = match['score']
    curr_wkts = match['wickets']
    curr_target = match['target']
    base_prob = match['probability']

    scenarios = [
        {"name": "🔥 Explosive Over (+16 runs, 0 wkts)", "runs": 16, "wkts": 0, "desc": "Two boundaries & smart running"},
        {"name": "⚡ Positive Over (+10 runs, 0 wkts)", "runs": 10, "wkts": 0, "desc": "One boundary & singles"},
        {"name": "⚙️ Par Over (+7 runs, 0 wkts)", "runs": 7, "wkts": 0, "desc": "Only singles and twos"},
        {"name": "🛑 Tight Over (+3 runs, 0 wkts)", "runs": 3, "wkts": 0, "desc": "Great bowling, mostly dot balls"},
        {"name": "⚡ Strike Over (+5 runs, 1 wkt)", "runs": 5, "wkts": 1, "desc": "Wicket falls, momentum interrupted"},
        {"name": "🌪 Double Strike (+3 runs, 2 wkts)", "runs": 3, "wkts": 2, "desc": "Catastrophic over, 2 dismissals"}
    ]

    scenario_results = []
    for sc in scenarios:
        new_b = min(120, curr_balls + 6)
        new_s = curr_score + sc["runs"]
        new_w = min(10, curr_wkts + sc["wkts"])
        res = calculate_match_state(new_s, new_w, new_b, curr_target)
        delta = res['probability'] - base_prob
        scenario_results.append({
            "scenario": sc["name"],
            "desc": sc["desc"],
            "score": f"{new_s}/{new_w}",
            "prob": res['probability'],
            "delta": delta,
            "rrr": res['required_rr']
        })

    w_col1, w_col2 = st.columns([1.4, 1.0], gap="large")

    with w_col1:
        for item in scenario_results:
            d_color = "#10b981" if item["delta"] >= 0 else "#f43f5e"
            d_sign = "+" if item["delta"] >= 0 else ""
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.75); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 22px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:700; font-size:16px; color:#f8fafc; margin-bottom:3px;">{item['scenario']}</div>
                    <div style="color:#94a3b8; font-size:12px;">{item['desc']} · Projected: {item['score']} · Next RRR: {item['rrr']:.2f}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Space Grotesk', sans-serif; font-size:22px; font-weight:800; color:#f8fafc;">{item['prob']:.1f}%</div>
                    <div style="font-weight:700; font-size:13px; color:{d_color};">{d_sign}{item['delta']:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with w_col2:
        with st.container(border=True):
            st.markdown("#### 📈 Scenario Impact Comparison")
            names = [s["scenario"].split()[0] + " " + s["scenario"].split()[1] for s in scenario_results]
            probs = [s["prob"] for s in scenario_results]

            fig_bar = go.Figure(go.Bar(
                x=probs,
                y=names,
                orientation="h",
                marker=dict(
                    color=["#10b981", "#34d399", "#38bdf8", "#f59e0b", "#f97316", "#ef4444"],
                    line=dict(color="rgba(255,255,255,0.15)", width=1)
                ),
                text=[f"{p:.1f}%" for p in probs],
                textposition="inside",
                textfont=dict(color="#ffffff", size=12, family="Space Grotesk")
            ))
            fig_bar.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                xaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.06)", title="Win Probability (%)"),
                yaxis=dict(autorange="reversed"),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            st.caption("Strategic note: Wickets carry ~2.5x more negative weight on win probability than dot balls during death overs.")


def render_ai_chat_panel(match, ollama_online):
    """Renders the self-contained, scroll-locked right-side AI chatbox."""
    with st.container(border=True):
        # Header bar with close button
        h1, h2 = st.columns([2.8, 1.2])
        with h1:
            status_text = "🟢 Ollama Active" if ollama_online else "🟡 Offline Tactical"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:22px;">🤖</span>
                <div>
                    <div style="font-weight:800; font-size:16px; color:#ffffff; font-family:'Outfit', sans-serif;">Boundary AI Strategist</div>
                    <div style="font-size:11px; color:#10b981; font-weight:700;">{status_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h2:
            if st.button("✕ Close", key="btn_close_side_chat", use_container_width=True):
                st.session_state.show_ai_chat = False
                if st.session_state.nav_tab == "🤖 AI Cricket Strategist":
                    st.session_state.nav_tab = "🏟️ Match Center"
                st.rerun()

        # Prompt chips
        st.markdown("<div style='font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; margin:8px 0 6px;'>⚡ Quick Prompts</div>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("🏏 Batting", use_container_width=True, key="chip_batting_btn"):
                st.session_state.pending_prompt = f"What is the optimal batting approach for {st.session_state.team_a} given the current RRR of {match['required_rr']:.2f} and {match['wickets']} wickets down?"
                st.rerun()
        with q2:
            if st.button("🛡️ Bowling", use_container_width=True, key="chip_bowling_btn"):
                st.session_state.pending_prompt = f"How should {st.session_state.team_b} set their fields and rotate bowlers during this {match['phase']}?"
                st.rerun()
        with q3:
            if st.button("📊 Win %", use_container_width=True, key="chip_winprob_btn"):
                st.session_state.pending_prompt = f"Explain why the win probability is estimated at {match['probability']:.1f}% based on match situation and historical T20 models."
                st.rerun()

        # Inner Scrollable Chat History Container (Fixed height: 450px)
        # Prevents page-stretch and eliminates page scrolling!
        with st.container(height=450, border=True):
            if not st.session_state.chat_history:
                st.markdown(f"""
                <div style="text-align:center; padding:35px 12px; color:#94a3b8;">
                    <div style="font-size:36px; margin-bottom:8px;">🏏</div>
                    <div style="font-weight:700; color:#f8fafc; font-size:15px;">Boundary AI Strategist Ready</div>
                    <div style="font-size:12px; margin-top:6px; color:#64748b;">
                        Live Match Attached: <strong>{match['score']}/{match['wickets']}</strong> ({match['overs_str']} ov) · Need {match['runs_remaining']} off {match['balls_remaining']}b
                    </div>
                    <div style="font-size:11px; margin-top:10px; color:#94a3b8;">Ask tactical questions or click any quick inquiry above.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # Inline Chat Input beneath the scroll container
        user_input = st.chat_input("Ask tactics, bowling plans, rules...", key="side_panel_chat_input")

        # Handle pending prompt
        if st.session_state.get("pending_prompt"):
            user_input = st.session_state.pending_prompt
            del st.session_state.pending_prompt


        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # 1. Historical IPL questions use verified local dataset facts.
            historical_result = answer_historical_question(user_input)

            if historical_result["handled"]:
                factual_context = historical_result["answer"]

                if ollama_online:
                    historical_system_msg = f"""
You are Boundary AI, an IPL cricket analyst.

Explain the supplied historical data clearly and concisely.
The verified dataset facts are authoritative. Do not change
teams, dates, scores, overs, or wickets. If clarification is
requested, ask for it instead of guessing.
For over-by-over questions, include:
- Total runs scored in the over
- Batter runs and extras
- Wickets taken
- Notable boundaries, such as fours and sixes, when provided
- Delivery-by-delivery details when useful

Use only the verified dataset facts. Never invent missing details.
If a requested statistic is not available, say so clearly.
VERIFIED DATASET FACTS:
{factual_context}
"""
                    try:
                        resp = requests.post(
                            OLLAMA_URL,
                            json={
                                "model": OLLAMA_MODEL,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": historical_system_msg,
                                    },
                                    {
                                        "role": "user",
                                        "content": user_input,
                                    },
                                ],
                                "stream": False,
                            },
                            timeout=45,
                        )
                        resp.raise_for_status()

                        ai_text = (
                            resp.json()
                            .get("message", {})
                            .get("content", "")
                            .strip()
                        )

                        if (
                            not ai_text
                            or len(ai_text) < 100
                            or "Runs in over:" in factual_context
                            and (
                                 "Batter runs:" not in ai_text
                                 or "Extras:" not in ai_text
                                 or "Wickets in over:" not in ai_text
                                 )
                        ):
                            ai_text = factual_context

                    except Exception:
                        # Keep historical answers available if Qwen fails.
                        ai_text = factual_context
                else:
                    # Historical retrieval does not require Ollama.
                    ai_text = factual_context
                ai_text = format_historical_answer(ai_text)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ai_text}
                )

            # 2. Non-historical questions use Ollama when available.
            elif ollama_online:
                system_msg = f"""
You are Boundary AI, an elite cricket strategist and
tactical intelligence assistant.

Discuss match tactics, batting pacing, bowler match-ups,
field sets, T20 pressure dynamics, and ML predictions.

{st.session_state.last_match_context}

Be clear, realistic, and concise. Use bullet points where appropriate.
"""
                msgs = [{"role": "system", "content": system_msg}]
                msgs.extend(st.session_state.chat_history[-10:])

                try:
                    resp = requests.post(
                        OLLAMA_URL,
                        json={
                            "model": OLLAMA_MODEL,
                            "messages": msgs,
                            "stream": False,
                        },
                        timeout=45,
                    )
                    resp.raise_for_status()

                    ai_text = (
                        resp.json()
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )

                    if not ai_text:
                        ai_text = "Sorry, I couldn't generate a response."

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": ai_text}
                    )

                except Exception as e:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ Ollama error: {e}",
                        }
                    )

            # 3. General tactical fallback when Ollama is offline.
            else:
                tactical_answer = f"""
**Tactical Assessment**

- **Win Probability:** {match['probability']:.1f}%
- **Current Score:** {match['score']}/{match['wickets']}
- **Target:** {match['target']}
- **Runs Required:** {match['runs_remaining']}
- **Balls Remaining:** {match['balls_remaining']}
- **Required Run Rate:** {match['required_rr']:.2f}

**Batting Team — {st.session_state.team_a}**

Aim to rotate the strike and target boundaries according
to the required run rate.

**Bowling Team — {st.session_state.team_b}**

Focus on limiting boundaries, varying pace, and executing
consistent line and length.
"""
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": tactical_answer}
                )

            st.rerun()

        # Panel Footer Actions
        f1, f2 = st.columns([1, 1])
        with f1:
            if st.session_state.chat_history:
                if st.button("🗑️ Clear History", key="btn_clear_drawer_chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        with f2:
            st.caption(f"Context: {match['score']}/{match['wickets']} · {match['runs_remaining']} off {match['balls_remaining']}b")


# =========================================================
# DYNAMIC LAYOUT DISPATCHER
# =========================================================
if st.session_state.show_ai_chat:
    # Side-by-side mode: Main Dashboard on Left + AI Chatbox on Right
    col_main, col_chat = st.columns([1.6, 1.1], gap="large")

    with col_main:
        if active_tab == "⚡ Live Ball Simulator":
            render_ball_simulator_content(match)
        elif active_tab == "🔮 What-If Scenario Matrix":
            render_what_if_matrix_content(match)
        else: # "🏟️ Match Center" or "🤖 AI Cricket Strategist"
            render_match_center_content(match, team_a_info, team_b_info)

    with col_chat:
        render_ai_chat_panel(match, ollama_online)

else:
    # Full-width mode: Primary analytics displayed without chat sidebar
    if active_tab == "⚡ Live Ball Simulator":
        render_ball_simulator_content(match)
    elif active_tab == "🔮 What-If Scenario Matrix":
        render_what_if_matrix_content(match)
    elif active_tab == "🤖 AI Cricket Strategist":
        # Automatically toggle open the side drawer when user selects AI tab
        st.session_state.show_ai_chat = True
        st.rerun()
    else:
        render_match_center_content(match, team_a_info, team_b_info)


# =========================================================
# FOOTER
# =========================================================
st.write("")
st.divider()
st.markdown("""
<div style="text-align:center; padding:15px; color:#64748b; font-size:12px;">
    <strong>BOUNDARY</strong> · Sports Intelligence Platform · Calibrated Random Forest Classification + Local AI Engine
    <br>Predictions reflect statistical historical win probabilities based on live in-game match vectors.
</div>
""", unsafe_allow_html=True)
