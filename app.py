
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import sys
import os
import gdown

sys.path.append("src")
from model import AttentionUNet

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Segmentation")
st.markdown("### Attention U-Net | Deep Learning | IITP")
st.markdown("---")

with st.sidebar:
    st.header("📊 Model Info")
    st.metric("Architecture", "Attention U-Net")
    st.metric("Test Dice",    "0.8207")
    st.metric("Test IoU",     "0.7903")
    st.metric("Parameters",   "31.4M")
    st.metric("Dataset",      "LGG Brain MRI (110 patients)")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a Brain MRI (.tif / .png)")
    st.markdown("2. Attention U-Net segments tumor")
    st.markdown("3. View mask + overlay + heatmap")
    threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.05)

# ── Load Model ─────────────────────────────────────────────────
MODEL_PATH = "checkpoints/best_model.pth"
GDRIVE_ID  = "1PIlPPPrzvRWD6QX-YZdDcHxT_vAnVErD"

@st.cache_resource
def load_model():
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading model weights (~120MB)..."):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_ID}",
                MODEL_PATH, quiet=False
            )
    device = torch.device("cpu")
    model  = AttentionUNet()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()
    return model, device

model, device = load_model()

# ── Predict ────────────────────────────────────────────────────
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

# ── Upload ─────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Brain MRI Image (.tif / .png / .jpg)",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np  = np.array(image_pil)

    with st.spinner("🔍 Running segmentation..."):
        mask, prob_map = predict(image_np, threshold)

    display_img  = cv2.resize(image_np, (256, 256))
    mask_display = (mask * 255).astype(np.uint8)

    overlay = display_img.copy()
    overlay[mask == 1] = [255, 50, 50]
    blended = cv2.addWeighted(display_img, 0.6, overlay, 0.4, 0)

    coverage  = round(mask.mean() * 100, 2)
    has_tumor = coverage > 0.5

    st.markdown("### 🔬 Segmentation Results")
    col1, col2, col3 = st.columns(3)
    col1.image(display_img,  caption="Brain MRI Input",      use_column_width=True)
    col2.image(mask_display, caption="Predicted Tumor Mask", use_column_width=True, clamp=True)
    col3.image(blended,      caption="Overlay (Red=Tumor)",  use_column_width=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tumor Detected", "✅ Yes" if has_tumor else "❌ No")
    m2.metric("Tumor Coverage", f"{coverage}%")
    m3.metric("Max Confidence", f"{prob_map.max()*100:.1f}%")

    st.markdown("### 🌡️ Confidence Heatmap")
    heatmap = cv2.applyColorMap(
        (prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
             caption="Model confidence (blue=low, red=high)",
             width=320)

else:
    st.info("👆 Upload a Brain MRI scan to get started")
    st.markdown("""
    **Sample images to test with:**  
    Download any `.tif` file from the dataset:  
    `data/raw/kaggle_3m/TCGA_CS_XXXX/` in your Google Drive
    """)
