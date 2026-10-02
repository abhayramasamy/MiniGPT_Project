import time
import requests
import streamlit as st
import plotly.graph_objects as go
from collections import deque

st.set_page_config(page_title="miniGPT // Observability", layout="wide", initial_sidebar_state="collapsed")

BG = "#0A0E14"
PANEL = "#10151D"
BORDER = "#1C242F"
TEXT = "#E6EDF3"
MUTED = "#6B7785"
MINT = "#00E8A0"
AMBER = "#FFB020"
RED = "#FF4D4D"
CYAN = "#3DD6F5"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    #MainMenu, footer, header {{visibility: hidden;}}

    [data-testid="stSidebar"] {{
        background-color: {PANEL};
    }}

    .mono {{
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
    }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 20px;
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        margin-bottom: 18px;
    }}
    .topbar-title {{
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    .topbar-sub {{
        color: {MUTED};
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 10px;
    }}
    .status-pill {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}
    .status-live {{
        background: rgba(0,232,160,0.1);
        color: {MINT};
        border: 1px solid rgba(0,232,160,0.3);
    }}
    .status-down {{
        background: rgba(255,77,77,0.1);
        color: {RED};
        border: 1px solid rgba(255,77,77,0.3);
    }}
    .dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: currentColor;
        display: inline-block;
    }}

    .card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
    }}
    .card-label {{
        color: {MUTED};
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .card-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: {TEXT};
        line-height: 1;
    }}
    .card-unit {{
        font-size: 14px;
        color: {MUTED};
        font-weight: 400;
    }}
    .badge-nominal {{ color: {MINT}; font-size: 10px; font-family: 'JetBrains Mono', monospace; }}
    .badge-watch {{ color: {AMBER}; font-size: 10px; font-family: 'JetBrains Mono', monospace; }}

    .panel-title {{
        color: {MUTED};
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}
    .detail-row {{
        display: flex;
        justify-content: space-between;
        padding: 7px 0;
        border-bottom: 1px solid {BORDER};
        font-size: 13px;
    }}
    .detail-row:last-child {{ border-bottom: none; }}
    .detail-label {{ color: {MUTED}; }}
    .detail-value {{ font-family: 'JetBrains Mono', monospace; color: {TEXT}; }}
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("**Connection**")
    api_url = st.text_input("API base URL", "http://localhost:5000")
    refresh_sec = st.slider("Refresh interval (s)", 1, 15, 3)
def fetch(path):
    try:
        r = requests.get(f"{api_url}{path}", timeout=4)
        r.raise_for_status()
        return r.json(), True
    except Exception:
        return {}, False

health, health_ok = fetch("/dev/health")
stats, stats_ok = fetch("/dev/stats")
server_up = health_ok and stats_ok

if "hist_tps" not in st.session_state:
    st.session_state.hist_tps = deque(maxlen=40)
    st.session_state.hist_latency = deque(maxlen=40)
    st.session_state.hist_requests = deque(maxlen=40)

if server_up:
    st.session_state.hist_tps.append(stats.get("average_tokens_per_second", 0))
    st.session_state.hist_latency.append(stats.get("average_generation_latency_ms", 0))
    st.session_state.hist_requests.append(stats.get("requests_total", 0))

def sparkline(data, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(data), mode="lines",
        line=dict(color=color, width=1.6),
        fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba"),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=46,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

def gauge(score):
    color = MINT if score >= 70 else (AMBER if score >= 40 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 34, "color": TEXT, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickwidth": 0, "showticklabels": False},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(255,77,77,0.12)"},
                {"range": [40, 70], "color": "rgba(255,176,32,0.12)"},
                {"range": [70, 100], "color": "rgba(0,232,160,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        height=190, margin=dict(l=20, r=20, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT},
    )
    return fig

def composite_score():
    if not server_up:
        return 0
    err_total = stats.get("errors_total", 0)
    req_total = max(stats.get("requests_total", 1), 1)
    error_rate = err_total / req_total
    score = 100
    score -= min(error_rate * 300, 60)          # errors hurt a lot
    if stats.get("active_requests", 0) > 8:
        score -= 10                              # heavy concurrent load
    return max(0, min(100, round(score)))

status_html = (
    f'<span class="status-pill status-live"><span class="dot"></span>PRODUCTION LIVE</span>'
    if server_up else
    f'<span class="status-pill status-down"><span class="dot"></span>UNREACHABLE</span>'
)
model_name = health.get("model", "miniGPT") if server_up else "miniGPT"
st.markdown(f"""
<div class="topbar">
    <div>
        <span class="topbar-title">miniGPT // OBSERVABILITY</span>
        <span class="topbar-sub">{model_name}</span>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
        <span class="topbar-sub">last refresh {time.strftime('%H:%M:%S UTC', time.gmtime())}</span>
        {status_html}
    </div>
</div>
""", unsafe_allow_html=True)

if not server_up:
    st.warning(f"Could not reach {api_url} — confirm the server is running and the URL is correct.")

col_left, col_right = st.columns([2.2, 1])

with col_left:
    r1 = st.columns(3)
    metrics = [
        ("TOKENS / SEC", stats.get("average_tokens_per_second", 0), "", st.session_state.hist_tps, MINT),
        ("GEN LATENCY", stats.get("average_generation_latency_ms", 0), "ms", st.session_state.hist_latency, CYAN),
        ("TIME TO FIRST TOKEN", stats.get("average_time_to_first_token_ms", 0), "ms", st.session_state.hist_latency, AMBER),
    ]
    for col, (label, value, unit, hist, color) in zip(r1, metrics):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value} <span class="card-unit">{unit}</span></div>
                <div class="badge-nominal">● NOMINAL</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(sparkline(hist, color), use_container_width=True, config={"displayModeBar": False})

    r2 = st.columns(3)
    err_total = stats.get("errors_total", 0)
    req_total = max(stats.get("requests_total", 0), 1)
    error_rate = round((err_total / req_total) * 100, 2) if server_up else 0
    secondary = [
        ("ACTIVE REQUESTS", stats.get("active_requests", 0), "", MINT),
        ("ERROR RATE", error_rate, "%", RED if error_rate > 5 else MINT),
        ("TOKENS GENERATED", stats.get("total_tokens_generated", 0), "total", CYAN),
    ]
    for col, (label, value, unit, color) in zip(r2, secondary):
        with col:
            badge = "WATCH" if (label == "ERROR RATE" and error_rate > 5) else "NOMINAL"
            badge_class = "badge-watch" if badge == "WATCH" else "badge-nominal"
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value} <span class="card-unit">{unit}</span></div>
                <div class="{badge_class}">● {badge}</div>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Composite Score</div>', unsafe_allow_html=True)
    st.plotly_chart(gauge(composite_score()), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div style="text-align:center; color:{MUTED}; font-size:12px; font-family:\'JetBrains Mono\',monospace; margin-top:-10px;">HEALTHY / N/A</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Deployment Details</div>', unsafe_allow_html=True)
    rows = [
        ("Model", health.get("model", "—")),
        ("Device", health.get("device", "—")),
        ("Dtype", health.get("dtype", "—")),
        ("Context length", health.get("context_length", "—")),
        ("Parameters", f"{health.get('parameter_count', 0):,}" if health.get("parameter_count") else "—"),
        ("Uptime", f"{health.get('uptime_seconds', 0):,.0f}s"),
    ]
    for label, value in rows:
        st.markdown(f"""
        <div class="detail-row">
            <span class="detail-label">{label}</span>
            <span class="detail-value">{value}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
with st.expander("Test generation", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        prompt = st.text_input("Prompt", "A wise crow")
    with c2:
        max_new = st.slider("Max new tokens", 5, 200, 50)
    c3, c4 = st.columns(2)
    with c3:
        temp = st.slider("Temperature", 0.1, 1.5, 0.7)
    with c4:
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9)

    if st.button("Generate", type="primary"):
        with st.spinner("Generating..."):
            try:
                resp = requests.post(
                    f"{api_url}/dev/generate",
                    json={"prompt": prompt, "max_new_tokens": max_new, "temperature": temp, "top_p": top_p},
                    timeout=60,
                ).json()
                st.write(resp.get("text", "(no output)"))
                st.json(resp.get("metrics", {}))
            except Exception as e:
                st.error(f"Request failed: {e}")
time.sleep(refresh_sec)
st.rerun()