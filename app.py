# file: app_realtime.py
# Streamlit app para detección de grietas en video casi en tiempo real
# usando streamlit-webrtc y un modelo BiSeNetV2 binario.
# La app recibe frames de la cámara del navegador, ejecuta inferencia
# y devuelve el overlay o la máscara binaria.

import os
import threading

import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from models.bisenetv2_binary import build_model


# =========================================================
# CONFIG
# =========================================================
IMG_SIZE = 512
THRESHOLD_DEFAULT = 0.7
ALPHA_DEFAULT = 0.35
MODEL_PATH = "best_bisenetv2_crack.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos: {MODEL_PATH}"
        )

    model = build_model(num_classes=1).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


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
    overlay[mask_255 > 0] = (0, 0, 255)   # rojo en BGR
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0)
    return blended


# =========================================================
# PROCESADOR DE VIDEO
# =========================================================
class VideoProcessor:
    def __init__(self):
        self.model = load_model()
        self.lock = threading.Lock()

        self.threshold = THRESHOLD_DEFAULT
        self.alpha = ALPHA_DEFAULT
        self.view_mode = "overlay"   # overlay | mask | original

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            threshold = self.threshold
            alpha = self.alpha
            view_mode = self.view_mode

        mask = predict_mask(self.model, img, threshold)

        if view_mode == "mask":
            out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif view_mode == "original":
            out = img
        else:
            out = make_overlay(img, mask, alpha)

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# =========================================================
# APP
# =========================================================
def main():
    st.set_page_config(page_title="Grietas en tiempo real", layout="wide")
    st.title("Detección de grietas en tiempo real con BiSeNetV2")

    st.write(
        "Pulsa Start, permite acceso a la cámara y la app mostrará el overlay "
        "sobre el video casi en tiempo real."
    )

    st.info(
        f"Dispositivo usado por el modelo: {DEVICE}. "
        "En CPU puede funcionar más lento."
    )

    with st.sidebar:
        st.header("Parámetros")
        threshold = st.slider("Threshold", 0.1, 0.95, THRESHOLD_DEFAULT, 0.05)
        alpha = st.slider("Alpha overlay", 0.05, 0.95, ALPHA_DEFAULT, 0.05)
        view_mode = st.selectbox(
            "Visualización",
            ["overlay", "mask", "original"],
            index=0
        )

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key="crack-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"}
            },
            "audio": False,
        },
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        with webrtc_ctx.video_processor.lock:
            webrtc_ctx.video_processor.threshold = threshold
            webrtc_ctx.video_processor.alpha = alpha
            webrtc_ctx.video_processor.view_mode = view_mode

    st.markdown(
        """
        Uso:
        1. Pulsa Start.
        2. Acepta permisos de cámara en el navegador.
        3. Ajusta threshold y alpha si lo necesitas.
        4. Para usar cámara trasera en teléfono/tablet, el navegador suele elegirla por facingMode=environment.
        """
    )


if __name__ == "__main__":
    main()