import time
from pathlib import Path
from datetime import datetime, timedelta
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Safran × AIonOS | Engine Acoustic Diagnostics", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 1rem;}
      .stMetric {padding: 6px 10px;}
      .tight-card {padding: 12px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,0.15); background: rgba(7,18,40,0.45);}
      .muted {opacity: 0.78;}
      .small {font-size: 0.92rem;}
      .hero {padding: 10px 0 4px 0;}
      .pill {display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(120,170,255,0.35); margin-right:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✈️🧠 Safran × AIonOS | Aircraft Engine Acoustic Diagnostics, Ops Control & Quality Check")
st.caption("Inline test-cell video, acoustic telemetry, AI diagnostic copilot, parametric identification, inspection scoring, and production management.")

VIDEO_PATH = Path("assets/videos/engine_test_cell.mp4")


def render_inline_video(video_path: Path, height: int = 360):
    if not video_path.exists():
        st.warning("Video file missing in assets/videos.")
        return
    video_bytes = video_path.read_bytes()
    import base64
    encoded = base64.b64encode(video_bytes).decode()
    video_html = f"""
    <div style='border:1px solid rgba(120,170,255,0.28);border-radius:16px;overflow:hidden;background:#030a16;'>
      <video id='engineVideo' width='100%' height='{height}' controls autoplay loop playsinline preload='auto' style='display:block;background:black;'>
        <source src='data:video/mp4;base64,{encoded}' type='video/mp4'>
      </video>
      <div style='padding:8px 12px;color:#b9c7e6;font-size:12px;'>
        Browser note: looping works inline. Autoplay with sound can be blocked until the first user interaction by browser policy.
      </div>
    </div>
    <script>
      const v = document.getElementById('engineVideo');
      if (v) {{
        v.loop = true;
        v.muted = false;
        const tryPlay = () => v.play().catch(() => {{}});
        tryPlay();
        document.addEventListener('click', tryPlay, {{once:true}});
      }}
    </script>
    """
    components.html(video_html, height=height + 40)


def make_engine_series(seed: int, n: int = 360, noise: float = 0.45, drift: float = 0.6, stress: float = 0.8):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    throttle = 0.58 + 0.18*np.sin(2*np.pi*t/110) + 0.08*np.sin(2*np.pi*t/27)
    throttle += stress * 0.06 * rng.normal(0, 1, n)
    throttle = np.clip(throttle, 0.25, 0.98)

    rpm = 6200 + 5200*throttle + rng.normal(0, noise*90, n)
    egt = 460 + 290*throttle + rng.normal(0, noise*6, n)
    oil_temp = 82 + 16*throttle + rng.normal(0, noise*1.5, n)

    resonance_drift = np.clip((drift*0.0019)*t, 0, 0.95)
    imbalance = np.clip((drift*0.0013)*t + 0.03*np.sin(2*np.pi*t/60), 0, 0.95)
    bearing_wear = np.clip((drift*0.0015)*t, 0, 0.95)

    for k in range(120, n, 130):
        resonance_drift[k:] -= 0.08
        imbalance[k:] -= 0.05
        bearing_wear[k:] -= 0.07

    resonance_drift = np.clip(resonance_drift, 0, 0.95)
    imbalance = np.clip(imbalance, 0, 0.95)
    bearing_wear = np.clip(bearing_wear, 0, 0.95)

    acoustic_rms = 72 + 10*throttle + 16*imbalance + 13*bearing_wear + rng.normal(0, noise*1.1, n)
    spectral_centroid = 1750 + 950*throttle + 550*resonance_drift + rng.normal(0, noise*18, n)
    resonance_peak_hz = 2480 + 420*resonance_drift + 120*imbalance + rng.normal(0, noise*10, n)
    defect_score = np.clip(18 + 52*bearing_wear + 34*imbalance + 18*resonance_drift + rng.normal(0, noise*2, n), 0, 100)
    quality_score = np.clip(99.2 - 4.8*bearing_wear - 2.8*imbalance - 1.6*resonance_drift + rng.normal(0, noise*0.18, n), 88, 99.8)
    inspection_pass_rate = np.clip(98.9 - 3.9*defect_score/100 + rng.normal(0, noise*0.15, n), 90, 99.8)
    oee = np.clip(91 - 8.5*defect_score/100 - 4.3*resonance_drift + 1.5*throttle + rng.normal(0, noise*0.25, n), 72, 96)
    throughput = np.clip(38 + 12*throttle - 8.0*defect_score/100 + rng.normal(0, noise*0.4, n), 20, 52)
    downtime_risk = np.clip(8 + 66*bearing_wear + 20*imbalance + 10*resonance_drift + rng.normal(0, noise*1.5, n), 0, 100)

    return {
        "t": t,
        "throttle": throttle,
        "rpm": rpm,
        "egt": egt,
        "oil_temp": oil_temp,
        "acoustic_rms": acoustic_rms,
        "spectral_centroid": spectral_centroid,
        "resonance_peak_hz": resonance_peak_hz,
        "resonance_drift": resonance_drift,
        "imbalance": imbalance,
        "bearing_wear": bearing_wear,
        "defect_score": defect_score,
        "quality_score": quality_score,
        "inspection_pass_rate": inspection_pass_rate,
        "oee": oee,
        "throughput": throughput,
        "downtime_risk": downtime_risk,
    }


def status_from_score(x: float):
    if x >= 70:
        return "ALERT"
    if x >= 40:
        return "WATCH"
    return "NORMAL"


def compute_scores(x, cursor):
    rms = float(x["acoustic_rms"][cursor])
    centroid = float(x["spectral_centroid"][cursor])
    peak = float(x["resonance_peak_hz"][cursor])
    resonance = float(x["resonance_drift"][cursor])
    imbalance = float(x["imbalance"][cursor])
    wear = float(x["bearing_wear"][cursor])
    defect = float(x["defect_score"][cursor])

    acoustic_bad = np.clip((rms - 79)/18, 0, 1)
    centroid_bad = np.clip((centroid - 2450)/900, 0, 1)
    peak_bad = np.clip((peak - 2660)/280, 0, 1)

    diagnostic_risk = float(np.clip((0.32*acoustic_bad + 0.28*centroid_bad + 0.22*peak_bad + 0.18*wear)*100, 0, 100))
    quality_risk = float(np.clip((0.46*defect/100 + 0.28*imbalance + 0.26*resonance)*100, 0, 100))
    ops_risk = float(np.clip((0.52*x["downtime_risk"][cursor]/100 + 0.30*(100-x["oee"][cursor])/100 + 0.18*(100-x["inspection_pass_rate"][cursor])/100)*100, 0, 100))

    rul = float(np.clip(220 * (1 - (max(diagnostic_risk, quality_risk, ops_risk)/100)**1.3), 8, 220))

    findings = []
    if wear > 0.48 and acoustic_bad > 0.45:
        findings.append("Bearing wear signature rising in the acoustic envelope.")
    if imbalance > 0.42:
        findings.append("Rotor imbalance indicated by harmonic lift and RMS growth.")
    if resonance > 0.38:
        findings.append("Resonance drift suggests modal shift versus baseline acoustic twin.")
    if defect > 55:
        findings.append("Quality check agent would escalate this unit for targeted inspection.")
    if x["oee"][cursor] < 84:
        findings.append("Production control should re-sequence work to protect line throughput.")
    if not findings:
        findings.append("Engine remains within monitored acoustic-control band.")

    return diagnostic_risk, quality_risk, ops_risk, rul, findings


def make_line(fig, x, y, name, yaxis=None):
    trace = go.Scatter(x=x, y=y, mode="lines", name=name)
    if yaxis:
        trace.update(yaxis=yaxis)
    fig.add_trace(trace)


def genbi_answer(q, x, cursor, diag, qual, ops, rul, findings):
    if not q:
        return None, None
    qq = q.lower().strip()
    if "risk" in qq or "status" in qq:
        return (
            f"Diagnostic risk is {diag:.0f}/100, quality risk is {qual:.0f}/100, ops risk is {ops:.0f}/100. "
            f"Current state: diagnostics={status_from_score(diag)}, quality={status_from_score(qual)}, operations={status_from_score(ops)}.",
            None,
        )
    if "rul" in qq or "maintenance" in qq or "service" in qq:
        next_dt = (datetime.now() + timedelta(hours=rul)).strftime("%d %b %Y, %I:%M %p")
        return f"Predicted intervention window is in ~{rul:.0f} hours. Suggested service slot: {next_dt}.", None
    if "root" in qq or "cause" in qq or "why" in qq:
        return "Likely causes: " + " ".join(findings[:3]), None
    m = re.search(r"last\s+(\d+)\s+(ticks|points).*(rms|rpm|egt|oee|quality|throughput)", qq)
    if m:
        n = int(m.group(1))
        key_word = m.group(3)
        mapping = {
            "rms": ("acoustic_rms", "Acoustic RMS", "dB"),
            "rpm": ("rpm", "RPM", "rpm"),
            "egt": ("egt", "EGT", "°C"),
            "oee": ("oee", "OEE", "%"),
            "quality": ("quality_score", "Quality Score", "%"),
            "throughput": ("throughput", "Throughput", "units/hr"),
        }
        key, title, unit = mapping[key_word]
        s = max(0, cursor - n)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x["t"][s:cursor+1], y=x[key][s:cursor+1], mode="lines", name=title))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Tick", yaxis_title=unit)
        return f"Showing last {cursor-s} ticks of {title}.", fig
    return "Try: current risk, root cause, next maintenance, or 'show last 80 ticks rms trend'.", None


with st.sidebar:
    st.header("Controls")
    autoplay = st.toggle("Autoplay telemetry", value=True)
    tick_ms = st.slider("Refresh speed (ms)", 150, 1500, 350, 10)
    st.divider()
    noise = st.slider("Sensor noise", 0.0, 3.0, 0.5, 0.1)
    drift = st.slider("Degradation drift", 0.0, 2.0, 0.7, 0.1)
    stress = st.slider("Load stress", 0.0, 2.0, 0.8, 0.1)
    st.divider()
    asset = st.selectbox("Engine asset", [
        "Test Cell A | LEAP-family Engine",
        "Test Cell B | Narrowbody Turbofan",
        "Assembly Line 2 | Final Acceptance Engine",
    ])
    st.divider()
    st.markdown("<span class='pill'>Acoustic Data Cloud</span><span class='pill'>AI Copilot</span><span class='pill'>Quality Agents</span>", unsafe_allow_html=True)

seed = abs(hash(asset)) % (10**6)
x = make_engine_series(seed, noise=noise, drift=drift, stress=stress)

if "cursor" not in st.session_state:
    st.session_state.cursor = 0
if "last_asset" not in st.session_state:
    st.session_state.last_asset = None
if st.session_state.last_asset != asset:
    st.session_state.last_asset = asset
    st.session_state.cursor = 0

cursor = int(np.clip(st.session_state.cursor, 0, len(x["t"]) - 1))
st.session_state.cursor = cursor

if autoplay:
    st.session_state.cursor = min(st.session_state.cursor + 2, len(x["t"]) - 1)
    time.sleep(tick_ms / 1000.0)
    st.rerun()

diag, qual, ops, rul_hours, findings = compute_scores(x, cursor)
next_maint_str = (datetime.now() + timedelta(hours=rul_hours)).strftime("%d %b %Y, %I:%M %p")
confidence = float(np.clip(74 + 14*x["bearing_wear"][cursor] + 10*x["resonance_drift"][cursor] - noise*3, 58, 94))

left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("🎥 Inline Engine Test-Cell Feed")
    st.write(f"**Asset:** {asset}")
    render_inline_video(VIDEO_PATH, height=360)

    st.markdown('<div class="tight-card">', unsafe_allow_html=True)
    st.markdown("### Executive Snapshot")
    a,b,c = st.columns(3)
    a.metric("Diagnostic State", status_from_score(diag))
    b.metric("Quality State", status_from_score(qual))
    c.metric("Operations State", status_from_score(ops))

    d,e,f = st.columns(3)
    d.metric("Diagnostic Risk", f"{diag:.0f}/100")
    e.metric("Inspection Pass", f"{x['inspection_pass_rate'][cursor]:.1f}%")
    f.metric("RUL", f"{rul_hours:.0f} hrs")

    st.markdown(f"**Predicted intervention window:** {next_maint_str}")
    st.markdown(f"<span class='muted'>Model confidence: {confidence:.0f}%</span>", unsafe_allow_html=True)
    st.markdown("#### Top findings")
    for item in findings[:5]:
        st.write(f"- {item}")

    st.markdown("#### GenBI quick query")
    quick_q = st.text_input("Ask about root cause, risk, or trends", placeholder="e.g., show last 80 ticks rms trend")
    st.markdown("</div>", unsafe_allow_html=True)

    qa, qfig = genbi_answer(quick_q, x, cursor, diag, qual, ops, rul_hours, findings) if quick_q else (None, None)
    if qa:
        st.info(qa)
    if qfig is not None:
        st.plotly_chart(qfig, use_container_width=True)

with right:
    st.subheader("📟 Acoustic AI Dashboard")
    r1,r2,r3 = st.columns(3)
    r1.metric("RPM", f"{x['rpm'][cursor]:.0f}")
    r2.metric("EGT", f"{x['egt'][cursor]:.1f} °C")
    r3.metric("Oil Temp", f"{x['oil_temp'][cursor]:.1f} °C")
    r4,r5,r6 = st.columns(3)
    r4.metric("Acoustic RMS", f"{x['acoustic_rms'][cursor]:.1f} dB")
    r5.metric("Spectral Centroid", f"{x['spectral_centroid'][cursor]:.0f} Hz")
    r6.metric("Resonance Peak", f"{x['resonance_peak_hz'][cursor]:.0f} Hz")
    r7,r8,r9 = st.columns(3)
    r7.metric("Quality Score", f"{x['quality_score'][cursor]:.1f}%")
    r8.metric("OEE", f"{x['oee'][cursor]:.1f}%")
    r9.metric("Throughput", f"{x['throughput'][cursor]:.1f}/hr")

    tabs = st.tabs(["📈 Live Telemetry", "🧠 Agent", "💬 GenBI Query"])

    with tabs[0]:
        window = 140
        start = max(0, cursor-window)
        fig = go.Figure()
        make_line(fig, x["t"][start:cursor+1], x["acoustic_rms"][start:cursor+1], "Acoustic RMS", None)
        make_line(fig, x["t"][start:cursor+1], x["defect_score"][start:cursor+1], "Defect Score", "y2")
        make_line(fig, x["t"][start:cursor+1], x["oee"][start:cursor+1], "OEE", "y3")
        fig.add_vline(x=x["t"][cursor], line_width=2)
        fig.update_layout(
            height=390,
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Telemetry Tick",
            yaxis=dict(title="Acoustic RMS (dB)"),
            yaxis2=dict(title="Defect Score", overlaying="y", side="right"),
            yaxis3=dict(title="OEE (%)", overlaying="y", side="right", position=0.97, showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns([1,2])
        with c1:
            if st.button("⏩ Advance telemetry"):
                st.session_state.cursor = min(st.session_state.cursor + 10, len(x["t"]) - 1)
                st.rerun()
        with c2:
            st.progress(int((cursor/(len(x["t"])-1))*100))

    with tabs[1]:
        c1,c2 = st.columns(2)
        with c1:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=rul_hours, number={"suffix":" hrs"}, gauge={"axis":{"range":[0,220]}, "bar":{"thickness":0.35}}, title={"text":"Remaining Useful Life"}))
            gauge.update_layout(height=280, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(gauge, use_container_width=True)
        with c2:
            s = max(0, cursor-140)
            diag_series, qual_series, ops_series = [], [], []
            for i in range(s, cursor+1):
                d,q,o,_,_ = compute_scores(x, i)
                diag_series.append(d)
                qual_series.append(q)
                ops_series.append(o)
            risk_fig = go.Figure()
            risk_fig.add_trace(go.Scatter(x=x["t"][s:cursor+1], y=diag_series, mode="lines", name="Diagnostic Risk"))
            risk_fig.add_trace(go.Scatter(x=x["t"][s:cursor+1], y=qual_series, mode="lines", name="Quality Risk"))
            risk_fig.add_trace(go.Scatter(x=x["t"][s:cursor+1], y=ops_series, mode="lines", name="Ops Risk"))
            risk_fig.add_hline(y=40, line_width=1)
            risk_fig.add_hline(y=70, line_width=1)
            risk_fig.add_vline(x=x["t"][cursor], line_width=2)
            risk_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Tick", yaxis_title="Risk (0-100)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(risk_fig, use_container_width=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Next Service Due By", next_maint_str)
        m2.metric("Confidence", f"{confidence:.0f}%")
        likely_issue = "Bearing wear" if x["bearing_wear"][cursor] > 0.5 else "Rotor imbalance" if x["imbalance"][cursor] > 0.45 else "Resonance drift" if x["resonance_drift"][cursor] > 0.4 else "No critical deviation"
        m3.metric("Most Likely Issue", likely_issue)
        if max(diag, qual, ops) >= 70:
            st.error("🚨 Recommendation: isolate the engine for diagnostic review, targeted inspection, and production-line containment.")
        elif max(diag, qual, ops) >= 40:
            st.warning("⚠️ Recommendation: continue controlled operation and schedule a focused acoustic-quality check in the next window.")
        else:
            st.success("✅ Recommendation: continue operation within baseline control band.")

    with tabs[2]:
        st.caption("Rule-based offline GenBI demo. Upgrade path to LLM + RAG is straightforward.")
        q = st.text_input("Your question", placeholder="What is the current risk and why?")
        ans, fig = genbi_answer(q, x, cursor, diag, qual, ops, rul_hours, findings) if q else (None, None)
        if ans:
            st.info(ans)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
