# file: app.py
'''
Streamlit app para segmentación binaria de grietas con BiSeNetV2.
Permite capturar una foto desde la cámara del teléfono/tablet o subir una imagen,
ejecutar inferencia y mostrar la máscara y el overlay.
'''

import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from models.bisenetv2_binary import build_model


# =========================
# CONFIG
# =========================
IMG_SIZE = 512
THRESHOLD_DEFAULT = 0.7
ALPHA_DEFAULT = 0.35
MODEL_FILENAME = "best_bisenetv2_crack.pth"

# En Streamlit Cloud normalmente correrás en CPU
DEVICE = "cpu"


# =========================
# UTILIDADES
# =========================
def resolve_model_path():
    candidates = [
        MODEL_FILENAME,
        os.path.join("outputs_bisenetv2", "checkpoints", MODEL_FILENAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No se encontró el archivo de pesos. "
        f"Se buscó en: {candidates}"
    )


@st.cache_resource
def load_model():
    model_path = resolve_model_path()
    model = build_model(num_classes=1).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, model_path


def pil_to_bgr(image_pil):
    rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def preprocess(frame_bgr, img_size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x).float().to(DEVICE)
    return x


def predict_mask(model, frame_bgr, threshold):
    h, w = frame_bgr.shape[:2]
    x = preprocess(frame_bgr, IMG_SIZE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    mask = (probs >= threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def make_overlay(frame_bgr, mask_255, alpha):
    overlay = frame_bgr.copy()
    overlay[mask_255 > 0] = (0, 0, 255)  # rojo en BGR
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)
    return blended


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def image_to_png_bytes(image_bgr):
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen PNG.")
    return buffer.tobytes()


# =========================
# APP
# =========================
def main():
    st.set_page_config(page_title="Detección de grietas", layout="wide")
    st.title("Detección de grietas con BiSeNetV2")

    st.write(
        "Puedes tomar una foto con la cámara del dispositivo o subir una imagen. "
        "La app genera una máscara binaria y un overlay de predicción."
    )

    model, model_path = load_model()
    st.caption(f"Pesos cargados desde: {model_path}")

    with st.sidebar:
        st.header("Parámetros")
        threshold = st.slider("Threshold", 0.1, 0.95, THRESHOLD_DEFAULT, 0.05)
        alpha = st.slider("Alpha overlay", 0.05, 0.95, ALPHA_DEFAULT, 0.05)

    tab1, tab2 = st.tabs(["Cámara", "Subir imagen"])

    image_pil = None

    with tab1:
        camera_file = st.camera_input("Tomar foto", key="camera_input_main")
        if camera_file is not None:
            image_pil = Image.open(camera_file)

    with tab2:
        uploaded_file = st.file_uploader(
            "Subir imagen",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="file_uploader_main"
        )
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file)

    if image_pil is not None:
        image_bgr = pil_to_bgr(image_pil)
        mask_255 = predict_mask(model, image_bgr, threshold=threshold)
        overlay_bgr = make_overlay(image_bgr, mask_255, alpha=alpha)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Imagen original")
            st.image(bgr_to_rgb(image_bgr), use_container_width=True)

        with col2:
            st.subheader("Máscara predicha")
            st.image(mask_255, clamp=True, use_container_width=True)

        with col3:
            st.subheader("Overlay")
            st.image(bgr_to_rgb(overlay_bgr), use_container_width=True)

        st.download_button(
            label="Descargar overlay PNG",
            data=image_to_png_bytes(overlay_bgr),
            file_name="overlay_prediccion.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()