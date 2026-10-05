import streamlit as st
from PIL import Image
import torch
import os
import torch.nn as nn
from torchvision import models, transforms

# DDIM imports
import core.logger as Logger
import model as Model
import core.metrics as Metrics

# ------------------------------------------------
# DEVICE SETUP
# ------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="NeuroFuse",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------------
# UI STYLE
# ------------------------------------------------
# st.markdown("""
# <style>
# .main-header {
#     background: linear-gradient(135deg, #1a1a2e, #0f3460);
#     padding: 2rem;
#     border-radius: 15px;
#     text-align: center;
#     color: white;
# }
# .result-card {
#     background: linear-gradient(135deg, #2ed573, #1e90ff);
#     padding: 1rem;
#     border-radius: 10px;
#     text-align: center;
#     color: white;
# }
# </style>
# """, unsafe_allow_html=True)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        border: 1px solid #e94560;
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    
    .image-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #444;
    }
    
    .gpu-badge {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #000;
        font-weight: bold;
    }
    .cpu-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa502 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #000;
        font-weight: bold;
    }
    
    .result-card {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------
# LOAD DDIM MODEL
# ------------------------------------------------
@st.cache_resource
def load_ddim_model():
    class Args:
        config = 'C:\\Users\\K. Kishan Babu\\Documents\\Diff-IF\\Diff-IF\\config\\diff-if-mif_val.json'
        phase = 'val'
        gpu_ids = None
        debug = False
        log_infer = False

    opt = Logger.parse(Args())
    opt = Logger.dict_to_nonedict(opt)

    diffusion = Model.create_model(opt)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'],
        schedule_phase='val'
    )

    diffusion.device = DEVICE
    return diffusion

# ------------------------------------------------
# LOAD CLASSIFIER
# ------------------------------------------------
@st.cache_resource
def load_classifier():
    model = models.convnext_base(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[2].in_features

    model.classifier = nn.Sequential(
        model.classifier[0],
        nn.Flatten(1),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )

    state = torch.load("best_convnext_Diff_if.pth", map_location=DEVICE)
    model.load_state_dict(state)

    model = model.to(DEVICE).eval()

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    return model, preprocess

# ------------------------------------------------
# PREPROCESS (MATCH DATASET EXACTLY)
# ------------------------------------------------
def resize_multiple_of_8(img):
    w, h = img.size
    return img.resize((w - w % 8, h - h % 8))

def preprocess_ddim(ct_path, mri_path):
    # Load images
    ct = Image.open(ct_path).convert("YCbCr")
    mri = Image.open(mri_path).convert("YCbCr")

    # Extract Y channel
    ct_y = ct.split()[0]
    mri_y = mri.split()[0]

    # Full RGB reference (IMPORTANT)
    img_full = Image.open(mri_path).convert("RGB")

    # Resize
    ct_y = resize_multiple_of_8(ct_y)
    mri_y = resize_multiple_of_8(mri_y)
    img_full = resize_multiple_of_8(img_full)

    # Normalize to [-1, 1]
    ct_tensor = transforms.ToTensor()(ct_y) * 2 - 1
    mri_tensor = transforms.ToTensor()(mri_y) * 2 - 1
    full_tensor = transforms.ToTensor()(img_full) * 2 - 1

    return {
        "ir": ct_tensor.unsqueeze(0).to(DEVICE),
        "vis": mri_tensor.unsqueeze(0).to(DEVICE),
        "img_full": full_tensor.unsqueeze(0).to(DEVICE)
    }

# ------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------
def classify_fused_image(model, preprocess, fused_img):
    import torch.nn.functional as F

    img = Image.fromarray(fused_img).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    prob = float(probs[0][1])
    label = "Tumor" if prob >= 0.5 else "Non-tumor"

    return label, prob

# ------------------------------------------------
# DATA LOADING
# ------------------------------------------------
def get_patient_list(base_dir):
    if not os.path.exists(base_dir):
        return []
    return sorted(os.listdir(base_dir))

def get_modalities(base_dir, patient):
    path = os.path.join(base_dir, patient)
    files = os.listdir(path)

    data = {}
    for f in files:
        if f.lower().startswith("ct"):
            data["CT"] = os.path.join(path, f)
        elif f.lower().startswith("mri"):
            data["MRI"] = os.path.join(path, f)
    return data

# ------------------------------------------------
# UI HEADER
# ------------------------------------------------
st.markdown("""
<div class="main-header">
<h1>🧠 NEUROFUSE</h1>
<p>CT-MRI Fusion + Tumor Classification</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f'<div class="gpu-badge">🚀 GPU: {gpu_name[:20]}...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cpu-badge">⚠️ CPU Mode</div>', unsafe_allow_html=True)
    
    st.divider()
# ------------------------------------------------
# MAIN
# ------------------------------------------------
input_dir = "input"
patients = get_patient_list(input_dir)

if not patients:
    st.error("No data found")
else:
    st.success(f"✅ Found {len(patients)} patients in dataset")
    patient = st.selectbox("Select Patient", patients)

    if patient:
        data = get_modalities(input_dir, patient)

        if "CT" in data and "MRI" in data:
            ct_path = data["CT"]
            mri_path = data["MRI"]

            col1, col2 = st.columns(2)

            st.markdown("### 📷 Source Images")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 CT Scan")
                ct_img = Image.open(ct_path)
                st.image(ct_img, use_container_width=True)
            with col2:
                st.markdown(f"#### 🟢 MRI")
                mri_img = Image.open(mri_path)
                st.image(mri_img, use_container_width=True)
                
                st.divider()

            if st.button("Run Fusion and Classification 🚀",type="primary", use_container_width=True):
                progress = st.progress(0)

                progress.progress(20)
                diffusion = load_ddim_model()

                progress.progress(40)
                val_data = preprocess_ddim(ct_path, mri_path)

                progress.progress(60)
                diffusion.feed_data(val_data)
                diffusion.test_ddim(continous=True)

                visuals = diffusion.get_val_current_visuals()

                # Fused luminance
                fake = Metrics.tensor2img(
                    visuals['DDPM'][-1].unsqueeze(0)
                )

                # RGB reference
                full = Metrics.tensor2img(visuals['img_full'])

                # Final fusion
                fused_img = Metrics.mergy_Y_RGB_to_YCbCr(fake, full)

                progress.progress(80)

                clf_model, clf_pre = load_classifier()
                label, _ = classify_fused_image(clf_model, clf_pre, fused_img)

                progress.progress(100)

                st.markdown('<div class="result-card">Fusion Complete</div>', unsafe_allow_html=True)

                st.markdown("### 🎯 Fusion Result")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("")    
                with col2:
                    st.markdown("#### 🌟 Fused Output")
                    st.image(fused_img, use_container_width=True)
                with col3:
                    st.write("") 

                color = "red" if label == "Tumor" else "green"
                st.markdown(
                    f"<h2 style='color:{color}; text-align:center'>{label}</h2>",
                    unsafe_allow_html=True
                )

                os.makedirs("output", exist_ok=True)
                save_path = f"output/{patient}.png"
                Image.fromarray(fused_img).save(save_path)

                st.success(f"Saved to {save_path}")
        else:
            st.warning("Need CT + MRI")
st.divider()
device_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
st.markdown(f"""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>Multi-Modality Image Fusion and Tumor Classification with Diffusion Model and ConvNeXt Classifier</p>
</div>
""", unsafe_allow_html=True)