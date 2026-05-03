import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import sys, os, gdown, base64, io

sys.path.append("src")
from model import AttentionUNet

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def img_to_b64(img_array):
    if img_array.ndim == 2:
        pil = Image.fromarray(img_array.astype(np.uint8), mode="L").convert("RGB")
    else:
        pil = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #030712 !important;
    color: #f1f5f9 !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1380px !important; }

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 900px 600px at 15% 5%,  rgba(6,182,212,0.07)  0%, transparent 70%),
        radial-gradient(ellipse 700px 500px at 85% 85%, rgba(124,58,237,0.06) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a1628 0%, #06101e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem !important; }

[data-testid="stFileUploader"] section {
    background: rgba(6,182,212,0.03) !important;
    border: 1.5px dashed rgba(6,182,212,0.18) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(6,182,212,0.4) !important;
    background: rgba(6,182,212,0.06) !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(6,182,212,0.1) !important;
    border: 1px solid rgba(6,182,212,0.2) !important;
    color: #06b6d4 !important;
    border-radius: 8px !important;
}
.stSpinner > div { border-top-color: #06b6d4 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 1.5rem;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#06b6d4,#7c3aed);
                        display:flex;align-items:center;justify-content:center;font-size:1.2rem;">🧠</div>
            <div>
                <div style="font-size:1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.02em;">NeuroScan AI</div>
                <div style="font-size:0.62rem;color:#475569;font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;">ATTENTION U-NET · v1.0</div>
            </div>
        </div>
    </div>
    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;color:#1e3a5f;font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">⬡ Model Performance</div>
    """ + "".join(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">{k}</span>
        <span style="font-size:0.75rem;font-weight:600;color:{c};font-family:'JetBrains Mono',monospace;">{v}</span>
    </div>""" for k,v,c in [
        ("Test Dice",    "0.8207",   "#06b6d4"),
        ("Test IoU",     "0.7903",   "#06b6d4"),
        ("Parameters",   "31.4 M",   "#7c3aed"),
        ("Architecture", "Att-UNet", "#7c3aed"),
        ("Best Epoch",   "44 / 50",  "#10b981"),
        ("Dataset",      "LGG MRI",  "#f59e0b"),
        ("Patients",     "110",      "#f59e0b"),
        ("MRI Slices",   "3,929",    "#f59e0b"),
        ("GPU",          "T4 Colab", "#94a3b8"),
    ]) + """
    <div style="margin-top:1.4rem;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#1e3a5f;font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">⬡ Pipeline</div>
    <div style="background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.1);
                border-radius:12px;padding:1rem;display:flex;flex-direction:column;gap:0.65rem;">
    """ + "".join(f"""
    <div style="display:flex;align-items:flex-start;gap:9px;">
        <div style="width:19px;height:19px;border-radius:50%;flex-shrink:0;
                    background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.25);
                    display:flex;align-items:center;justify-content:center;
                    font-size:0.58rem;color:#06b6d4;font-family:'JetBrains Mono',monospace;font-weight:600;">{n}</div>
        <div style="font-size:0.72rem;color:#64748b;line-height:1.55;">{t}</div>
    </div>""" for n,t in [
        ("1","Upload Brain MRI — .tif / .png / .jpg"),
        ("2","Attention gates isolate tumor region"),
        ("3","Segmentation mask generated in ms"),
        ("4","Metrics + heatmap displayed instantly"),
    ]) + "</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    threshold = st.slider("Detection Threshold", 0.05, 0.95, 0.50, 0.05)

# ── Model ──────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/best_model.pth"
GDRIVE_ID  = "1PIlPPPrzvRWD6QX-YZdDcHxT_vAnVErD"

@st.cache_resource
def load_model():
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}",
                           MODEL_PATH, quiet=False)
    device = torch.device("cpu")
    m = AttentionUNet()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m, device

model, device = load_model()
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def predict(image_array, thr=0.5):
    img    = cv2.resize(image_array, (256, 256))
    img    = img.astype(np.float32) / 255.0
    img    = (img - mean) / std
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))
    prob_np = prob.squeeze().cpu().numpy()
    return (prob_np > thr).astype(np.uint8), prob_np

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3.5rem 0 2.5rem;">
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.22);
                border-radius:50px;padding:6px 18px;margin-bottom:1.6rem;">
        <span style="width:7px;height:7px;border-radius:50%;background:#06b6d4;
                     box-shadow:0 0 10px #06b6d4;display:inline-block;"></span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                     color:#06b6d4;letter-spacing:0.12em;text-transform:uppercase;">
            AI-Powered Medical Imaging
        </span>
    </div>
    <h1 style="font-size:clamp(2.6rem,4.5vw,4.2rem);font-weight:800;line-height:1.08;
               letter-spacing:-0.04em;margin:0 0 1rem;
               background:linear-gradient(135deg,#ffffff 0%,#67e8f9 45%,#a78bfa 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        Brain Tumor<br>Segmentation
    </h1>
    <p style="font-size:0.95rem;color:#64748b;max-width:480px;margin:0 auto 2.8rem;line-height:1.8;">
        Attention U-Net trained on 110 LGG Brain MRI patients.
        Achieves <span style="color:#06b6d4;font-weight:600;">Dice 0.82</span> and
        <span style="color:#7c3aed;font-weight:600;">IoU 0.79</span> on test data.
    </p>
    <div style="display:inline-grid;grid-template-columns:repeat(4,1fr);
                background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                border-radius:18px;overflow:hidden;max-width:680px;width:100%;">
""" + "".join(f"""
        <div style="padding:1.1rem 1rem;{'border-right:1px solid rgba(255,255,255,0.07);' if i<3 else ''}">
            <div style="font-size:1.55rem;font-weight:700;color:{c};
                        font-family:'JetBrains Mono',monospace;line-height:1;">{v}</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">{k}</div>
        </div>""" for i,(k,v,c) in enumerate([
    ("Dice Score","0.8207","#06b6d4"),
    ("IoU Score", "0.7903","#7c3aed"),
    ("Parameters","31.4M", "#10b981"),
    ("MRI Slices","3,929", "#f59e0b"),
])) + """
    </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.2),rgba(124,58,237,0.2),transparent);margin:0 0 2.5rem;"></div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────
uploaded = st.file_uploader("", type=["tif","tiff","png","jpg","jpeg"])

if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 2rem;">
        <div style="font-size:0.75rem;color:#1e3a5f;font-family:'JetBrains Mono',monospace;">
            ↑ &nbsp; Drop a Brain MRI scan above to begin analysis
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:0.5rem;">
    """ + "".join(f"""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.4rem;">
        <div style="font-size:1.5rem;margin-bottom:0.6rem;">{icon}</div>
        <div style="font-size:0.85rem;font-weight:700;color:#f1f5f9;margin-bottom:0.4rem;">{title}</div>
        <div style="font-size:0.74rem;color:#475569;line-height:1.65;">{desc}</div>
    </div>""" for icon,title,desc in [
        ("⚡","Attention Gates",
         "Spatial attention suppresses irrelevant tissue and focuses the model on tumor boundaries."),
        ("🎯","Dice + BCE Loss",
         "Combined loss handles class imbalance between background and tumor voxels."),
        ("🔬","FLAIR MRI Modality",
         "Trained on FLAIR sequences from 110 LGG patients in The Cancer Genome Atlas."),
    ]) + "</div>", unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────
if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np  = np.array(image_pil)

    with st.spinner("Running Attention U-Net inference..."):
        mask, prob_map = predict(image_np, threshold)

    disp     = cv2.resize(image_np, (256, 256))
    mask_rgb = np.stack([mask*255]*3, axis=-1).astype(np.uint8)
    ov       = disp.copy(); ov[mask==1] = [0,220,255]
    blended  = cv2.addWeighted(disp, 0.5, ov, 0.5, 0)
    heatmap  = cv2.cvtColor(
        cv2.applyColorMap((prob_map*255).astype(np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB)

    coverage = round(mask.mean()*100, 2)
    conf     = round(float(prob_map.max())*100, 1)
    avg_conf = round(float(prob_map[mask==1].mean())*100,1) if mask.sum()>0 else 0.0
    detected = coverage > 0.5
    sc       = "#10b981" if detected else "#ef4444"
    st_txt   = "TUMOR DETECTED" if detected else "NO TUMOR FOUND"
    st_bg    = "rgba(16,185,129,0.1)" if detected else "rgba(239,68,68,0.1)"
    st_bdr   = "rgba(16,185,129,0.25)" if detected else "rgba(239,68,68,0.25)"

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:1rem 1.4rem;background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.07);border-radius:14px;margin-bottom:1.4rem;">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.05rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.02em;">
                Segmentation Analysis
            </span>
            <span style="background:{st_bg};border:1px solid {st_bdr};border-radius:50px;
                         padding:3px 12px;font-size:0.62rem;color:{sc};
                         font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;
                         text-transform:uppercase;">● {st_txt}</span>
        </div>
        <div style="font-size:0.65rem;color:#334155;font-family:'JetBrains Mono',monospace;">
            threshold={threshold:.2f} &nbsp;·&nbsp; 256×256 px
        </div>
    </div>
    """, unsafe_allow_html=True)

    def panel(b64, title, dot, label):
        return f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;overflow:hidden;">
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:9px 13px;border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:0.6rem;font-weight:600;color:#64748b;text-transform:uppercase;
                             letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;">{title}</span>
                <span style="width:7px;height:7px;border-radius:50%;background:{dot};
                             box-shadow:0 0 7px {dot};display:inline-block;"></span>
            </div>
            <img src="data:image/png;base64,{b64}"
                 style="width:100%;display:block;aspect-ratio:1/1;object-fit:cover;"/>
            <div style="padding:7px 13px;font-size:0.6rem;color:#334155;
                        font-family:'JetBrains Mono',monospace;
                        border-top:1px solid rgba(255,255,255,0.04);">{label}</div>
        </div>"""

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.4rem;">
        {panel(img_to_b64(disp),     "Input Scan",     "#06b6d4", "Original MRI")}
        {panel(img_to_b64(mask_rgb), "Tumor Mask",     "#7c3aed", "Binary prediction")}
        {panel(img_to_b64(blended),  "Overlay",        "#10b981", "Cyan = tumor region")}
        {panel(img_to_b64(heatmap),  "Confidence Map", "#f59e0b", "INFERNO colormap")}
    </div>
    """, unsafe_allow_html=True)

    def mcard(icon, val, name, color, bar_w, bar_col):
        return f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:1.3rem 1.4rem;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{bar_col};
                        border-radius:14px 14px 0 0;"></div>
            <div style="font-size:1.3rem;margin-bottom:0.5rem;">{icon}</div>
            <div style="font-size:1.9rem;font-weight:800;color:{color};
                        font-family:'JetBrains Mono',monospace;line-height:1;margin-bottom:4px;">{val}</div>
            <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;
                        font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">{name}</div>
            <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{bar_w}%;background:{bar_col};border-radius:3px;"></div>
            </div>
        </div>"""

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.4rem;">
        {mcard("🟢" if detected else "🔴", "YES" if detected else "NO",
               "Tumor Detected", sc, 100 if detected else 5,
               "linear-gradient(90deg,#10b981,#34d399)" if detected else "linear-gradient(90deg,#ef4444,#f87171)")}
        {mcard("📐", f"{coverage}%", "Tumor Coverage", "#06b6d4",
               min(coverage*5,100), "linear-gradient(90deg,#06b6d4,#22d3ee)")}
        {mcard("⚡", f"{conf}%",     "Max Confidence", "#7c3aed",
               conf, "linear-gradient(90deg,#7c3aed,#a78bfa)")}
        {mcard("📊", f"{avg_conf}%", "Avg Confidence", "#f59e0b",
               avg_conf, "linear-gradient(90deg,#f59e0b,#fcd34d)")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:260px 1fr;gap:1rem;">
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;overflow:hidden;">
            <div style="padding:9px 13px;border-bottom:1px solid rgba(255,255,255,0.05);
                        font-size:0.6rem;font-weight:600;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;">
                Probability Heatmap
            </div>
            <img src="data:image/png;base64,{img_to_b64(heatmap)}"
                 style="width:100%;display:block;"/>
            <div style="padding:8px 13px 10px;">
                <div style="height:5px;border-radius:4px;background:linear-gradient(90deg,
                    #000004,#3b0f6f,#8c2981,#de4968,#fe9f6d,#fcfdbf);margin-bottom:5px;"></div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.58rem;color:#334155;font-family:'JetBrains Mono',monospace;">Low</span>
                    <span style="font-size:0.58rem;color:#334155;font-family:'JetBrains Mono',monospace;">High</span>
                </div>
            </div>
        </div>
        <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:1.4rem;">
            <div style="font-size:0.6rem;font-weight:600;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
                Architecture & Training Details
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;">
            """ + "".join(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                            border-radius:10px;padding:0.85rem;">
                    <div style="font-size:0.65rem;color:{c};font-weight:600;
                                font-family:'JetBrains Mono',monospace;margin-bottom:3px;">{k}</div>
                    <div style="font-size:0.75rem;color:#94a3b8;line-height:1.5;">{v}</div>
                </div>""" for k,v,c in [
                ("Encoder",       "4× ConvBlock + MaxPool",       "#06b6d4"),
                ("Attention",     "Soft spatial gating",          "#06b6d4"),
                ("Bottleneck",    "1024 feature channels",         "#7c3aed"),
                ("Decoder",       "Upsample + skip connections",   "#7c3aed"),
                ("Loss",          "DiceLoss + BCEWithLogits",      "#10b981"),
                ("Optimizer",     "AdamW · lr=3e-4 · wd=1e-4",    "#10b981"),
                ("Scheduler",     "CosineAnnealingLR · T=50",     "#f59e0b"),
                ("Augmentation",  "Flip · Rotate · Elastic",      "#f59e0b"),
            ]) + """
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(6,182,212,0.15),rgba(124,58,237,0.15),transparent);margin:3rem 0 1.5rem;"></div>
<div style="display:flex;justify-content:space-between;padding-bottom:1rem;">
    <span style="font-size:0.62rem;color:#1e293b;font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;">
        NEUROSCAN AI · ATTENTION U-NET · IITP 2025
    </span>
    <span style="font-size:0.62rem;color:#1e293b;font-family:'JetBrains Mono',monospace;">
        DICE 0.8207 · IOU 0.7903 · 31.4M PARAMS
    </span>
</div>
""", unsafe_allow_html=True)