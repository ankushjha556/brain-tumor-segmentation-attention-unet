
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import sys, os, gdown, base64, io

sys.path.append("src")
from model import AttentionUNet

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #020817;
    color: #e2e8f0;
}

/* ── Hide Streamlit Defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Animated Background ── */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(6,182,212,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,92,246,0.05) 0%, transparent 60%),
        linear-gradient(180deg, #020817 0%, #0a1628 50%, #020817 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b35 0%, #091022 100%) !important;
    border-right: 1px solid rgba(6,182,212,0.15) !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

/* ── Hero Section ── */
.hero-wrapper {
    padding: 3rem 0 2rem 0;
    text-align: center;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #06b6d4;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #06b6d4;
    box-shadow: 0 0 8px #06b6d4;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(1.3); }
}
.hero-title {
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #06b6d4 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    max-width: 560px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
    font-weight: 400;
}

/* ── Stat Bar ── */
.stat-bar {
    display: flex;
    justify-content: center;
    gap: 0;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    overflow: hidden;
    max-width: 700px;
    margin: 0 auto 3rem;
    backdrop-filter: blur(10px);
}
.stat-item {
    flex: 1;
    padding: 1.2rem 1.5rem;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.06);
    transition: background 0.3s;
}
.stat-item:last-child { border-right: none; }
.stat-item:hover { background: rgba(6,182,212,0.05); }
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #06b6d4;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label {
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
}

/* ── Upload Zone ── */
.upload-section {
    background: linear-gradient(135deg,
        rgba(6,182,212,0.04) 0%,
        rgba(139,92,246,0.04) 100%);
    border: 1.5px dashed rgba(6,182,212,0.25);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.upload-section::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(6,182,212,0.03) 0%, transparent 70%);
    pointer-events: none;
}
.upload-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 0.4rem;
}
.upload-hint {
    font-size: 0.8rem;
    color: #475569;
    font-family: 'DM Mono', monospace;
}

/* ── Results Header ── */
.results-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.results-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
}
.results-pill {
    background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 50px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    color: #06b6d4;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Image Cards ── */
.img-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.3s ease;
}
.img-card:hover {
    border-color: rgba(6,182,212,0.2);
    box-shadow: 0 0 30px rgba(6,182,212,0.05);
    transform: translateY(-2px);
}
.img-card-header {
    padding: 0.75rem 1rem;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.img-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace;
}
.img-card-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}
.metric-card.green::before  { background: linear-gradient(90deg, #10b981, transparent); }
.metric-card.cyan::before   { background: linear-gradient(90deg, #06b6d4, transparent); }
.metric-card.violet::before { background: linear-gradient(90deg, #8b5cf6, transparent); }
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.metric-icon {
    font-size: 1.4rem;
    margin-bottom: 0.6rem;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-card.green  .metric-val  { color: #10b981; }
.metric-card.cyan   .metric-val  { color: #06b6d4; }
.metric-card.violet .metric-val  { color: #8b5cf6; }
.metric-name {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
}

/* ── Heatmap Section ── */
.heatmap-section {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}
.heatmap-title {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.heatmap-title::before {
    content: '';
    width: 12px; height: 2px;
    background: #8b5cf6;
    border-radius: 2px;
}

/* ── Confidence Bar ── */
.conf-bar-wrapper {
    margin-top: 1rem;
}
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: #475569;
    margin-bottom: 0.4rem;
}
.conf-bar-track {
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #06b6d4, #8b5cf6);
    transition: width 1s ease;
}

/* ── Sidebar Styling ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.5rem;
}
.sidebar-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #06b6d4, #8b5cf6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}
.sidebar-name {
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}
.sidebar-tag {
    font-size: 0.65rem;
    color: #475569;
    font-family: 'DM Mono', monospace;
}
.sidebar-section-title {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #334155;
    font-family: 'DM Mono', monospace;
    margin: 1.5rem 0 0.75rem 0;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.sidebar-stat-label {
    font-size: 0.78rem;
    color: #64748b;
    font-family: 'DM Mono', monospace;
}
.sidebar-stat-value {
    font-size: 0.85rem;
    font-weight: 600;
    color: #06b6d4;
    font-family: 'DM Mono', monospace;
}
.sidebar-info-box {
    background: rgba(6,182,212,0.05);
    border: 1px solid rgba(6,182,212,0.12);
    border-radius: 10px;
    padding: 0.9rem;
    margin-top: 1rem;
}
.sidebar-info-step {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
    font-size: 0.75rem;
    color: #94a3b8;
    line-height: 1.5;
}
.sidebar-info-step:last-child { margin-bottom: 0; }
.step-num {
    width: 18px; height: 18px;
    background: rgba(6,182,212,0.15);
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 50%;
    font-size: 0.6rem;
    color: #06b6d4;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: 'DM Mono', monospace;
    font-weight: 600;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] > div {
    background: rgba(6,182,212,0.04) !important;
    border: 1.5px dashed rgba(6,182,212,0.2) !important;
    border-radius: 14px !important;
}
.stSlider > div > div > div {
    background: linear-gradient(90deg, #06b6d4, #8b5cf6) !important;
}
[data-testid="stImage"] img {
    border-radius: 12px !important;
    width: 100% !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #06b6d4 !important; }

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%, rgba(6,182,212,0.2) 30%,
        rgba(139,92,246,0.2) 70%, transparent 100%);
    margin: 2rem 0;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.72rem;
    color: #1e293b;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🧠</div>
        <div>
            <div class="sidebar-name">NeuroScan AI</div>
            <div class="sidebar-tag">v1.0 · Attention U-Net</div>
        </div>
    </div>

    <div class="sidebar-section-title">Model Performance</div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Test Dice</span>
        <span class="sidebar-stat-value">0.8207</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Test IoU</span>
        <span class="sidebar-stat-value">0.7903</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Parameters</span>
        <span class="sidebar-stat-value">31.4M</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Architecture</span>
        <span class="sidebar-stat-value">Att-UNet</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Dataset</span>
        <span class="sidebar-stat-value">LGG MRI</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Training GPU</span>
        <span class="sidebar-stat-value">T4 Colab</span>
    </div>

    <div class="sidebar-section-title">How it Works</div>
    <div class="sidebar-info-box">
        <div class="sidebar-info-step">
            <div class="step-num">1</div>
            <span>Upload a Brain MRI scan (.tif / .png / .jpg)</span>
        </div>
        <div class="sidebar-info-step">
            <div class="step-num">2</div>
            <span>Attention gates focus on tumor regions</span>
        </div>
        <div class="sidebar-info-step">
            <div class="step-num">3</div>
            <span>View mask, overlay and confidence heatmap</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Detection Settings</div>',
                unsafe_allow_html=True)
    threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Lower = more sensitive. Higher = more precise.")

# ── Load Model ─────────────────────────────────────────────────
MODEL_PATH = "checkpoints/best_model.pth"
GDRIVE_ID  = "1PIlPPPrzvRWD6QX-YZdDcHxT_vAnVErD"

@st.cache_resource
def load_model():
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_ID}",
                MODEL_PATH, quiet=False
            )
    device = torch.device("cpu")
    model  = AttentionUNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def predict(image_array, threshold=0.5):
    img    = cv2.resize(image_array, (256, 256))
    img    = img.astype(np.float32) / 255.0
    img    = (img - mean) / std
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))
    prob_np = prob.squeeze().cpu().numpy()
    mask    = (prob_np > threshold).astype(np.uint8)
    return mask, prob_np

def apply_colormap(prob_map):
    heatmap = cv2.applyColorMap(
        (prob_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">AI-Powered Medical Imaging</div>
    <h1 class="hero-title">Brain Tumor<br>Segmentation</h1>
    <p class="hero-subtitle">
        Deep learning–powered tumor detection using Attention U-Net,
        trained on 110 LGG patients with a Dice score of 0.82.
    </p>
    <div class="stat-bar">
        <div class="stat-item">
            <div class="stat-value">0.8207</div>
            <div class="stat-label">Dice Score</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">0.7903</div>
            <div class="stat-label">IoU Score</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">31.4M</div>
            <div class="stat-label">Parameters</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3929</div>
            <div class="stat-label">MRI Slices</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Brain MRI Scan",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if not uploaded:
    st.markdown("""
    <div class="upload-section">
        <div style="font-size:2.5rem; margin-bottom:0.75rem;">🫧</div>
        <div class="upload-title">Drop your Brain MRI scan here</div>
        <div class="upload-hint">Supports .tif · .png · .jpg · .jpeg</div>
    </div>
    """, unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────
if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np  = np.array(image_pil)

    with st.spinner("Running Attention U-Net inference..."):
        mask, prob_map = predict(image_np, threshold)

    display_img  = cv2.resize(image_np, (256, 256))
    mask_display = (mask * 255).astype(np.uint8)
    overlay      = display_img.copy()
    overlay[mask == 1] = [0, 210, 255]
    blended  = cv2.addWeighted(display_img, 0.55, overlay, 0.45, 0)
    heatmap  = apply_colormap(prob_map)
    coverage = round(mask.mean() * 100, 2)
    conf     = round(float(prob_map.max()) * 100, 1)
    detected = coverage > 0.5

    # Results header
    status_col = "#10b981" if detected else "#ef4444"
    status_txt = "TUMOR DETECTED" if detected else "NO TUMOR FOUND"
    st.markdown(f"""
    <div class="results-header">
        <span class="results-title">Segmentation Analysis</span>
        <span class="results-pill" style="
            background: {'rgba(16,185,129,0.1)' if detected else 'rgba(239,68,68,0.1)'};
            border-color: {'rgba(16,185,129,0.3)' if detected else 'rgba(239,68,68,0.3)'};
            color: {status_col};">
            ● {status_txt}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Image grid
    c1, c2, c3, c4 = st.columns(4)
    panels = [
        (c1, display_img,          "INPUT SCAN",       "#06b6d4"),
        (c2, mask_display,         "TUMOR MASK",        "#8b5cf6"),
        (c3, blended,              "OVERLAY",           "#10b981"),
        (c4, heatmap,              "CONFIDENCE MAP",    "#f59e0b"),
    ]
    for col, img, title, dot_color in panels:
        with col:
            st.markdown(f"""
            <div class="img-card">
                <div class="img-card-header">
                    <span class="img-card-title">{title}</span>
                    <div class="img-card-dot"
                         style="background:{dot_color};
                                box-shadow: 0 0 6px {dot_color};">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(img, use_column_width=True,
                     clamp=True if title == "TUMOR MASK" else False)

    # Metric cards
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card {'green' if detected else 'metric-card-red'}">
            <div class="metric-icon">{"🟢" if detected else "🔴"}</div>
            <div class="metric-val" style="color:{'#10b981' if detected else '#ef4444'}">
                {"YES" if detected else "NO"}
            </div>
            <div class="metric-name">Tumor Detected</div>
        </div>
        <div class="metric-card cyan">
            <div class="metric-icon">📐</div>
            <div class="metric-val">{coverage}%</div>
            <div class="metric-name">Tumor Coverage</div>
            <div class="conf-bar-wrapper">
                <div class="conf-bar-track">
                    <div class="conf-bar-fill"
                         style="width:{min(coverage*4,100)}%;
                                background: linear-gradient(90deg,#06b6d4,#8b5cf6);">
                    </div>
                </div>
            </div>
        </div>
        <div class="metric-card violet">
            <div class="metric-icon">⚡</div>
            <div class="metric-val">{conf}%</div>
            <div class="metric-name">Max Confidence</div>
            <div class="conf-bar-wrapper">
                <div class="conf-bar-track">
                    <div class="conf-bar-fill"
                         style="width:{conf}%;
                                background: linear-gradient(90deg,#8b5cf6,#ec4899);">
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Heatmap deep view
    h1, h2 = st.columns([1, 2])
    with h1:
        st.markdown("""
        <div class="heatmap-section">
            <div class="heatmap-title">Probability Heatmap</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(heatmap, use_column_width=True)
    with h2:
        st.markdown("""
        <div class="heatmap-section" style="height:100%;">
            <div class="heatmap-title">Model Architecture</div>
            <div style="font-size:0.82rem; color:#64748b; line-height:1.9;
                        font-family:'DM Mono',monospace;">
                <div style="margin-bottom:0.4rem;">
                    <span style="color:#06b6d4;">Encoder</span>
                    → 4× ConvBlock + MaxPool
                </div>
                <div style="margin-bottom:0.4rem;">
                    <span style="color:#8b5cf6;">Attention Gates</span>
                    → Spatial focus on tumor
                </div>
                <div style="margin-bottom:0.4rem;">
                    <span style="color:#10b981;">Bottleneck</span>
                    → 1024 feature channels
                </div>
                <div style="margin-bottom:0.4rem;">
                    <span style="color:#f59e0b;">Decoder</span>
                    → Upsample + Skip connections
                </div>
                <div style="margin-bottom:0.4rem;">
                    <span style="color:#ec4899;">Loss</span>
                    → DiceLoss + BCEWithLogits
                </div>
                <div style="margin-bottom:0.4rem;">
                    <span style="color:#06b6d4;">Optimizer</span>
                    → AdamW · lr=3e-4
                </div>
                <div>
                    <span style="color:#8b5cf6;">Scheduler</span>
                    → CosineAnnealingLR
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="custom-divider"></div>
<div class="app-footer">
    NEUROSCAN AI · BUILT WITH ATTENTION U-NET · IITP · 2025
</div>
""", unsafe_allow_html=True)
