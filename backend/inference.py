"""
ONNX Runtime inference wrapper.
Loads nas_model.onnx once at startup, preprocesses images, runs inference.
"""

import io
import os
import time
import numpy as np
from PIL import Image

CIFAR10_CLASSES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3, 1, 1)

# Resolve ONNX path — prefer full-run exports, fall back to smoke
_BASE = os.path.dirname(os.path.dirname(__file__))
ONNX_PATHS = [
    os.path.join(_BASE, 'exports', 'nas_model.onnx'),
    os.path.join(_BASE, 'exports', 'smoke', 'nas_model.onnx'),
]

_session = None


def _get_session():
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")

    for path in ONNX_PATHS:
        if os.path.exists(path):
            _session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            print(f"ONNX model loaded from: {path}")
            return _session

    raise FileNotFoundError(
        f"No ONNX model found. Expected at: {ONNX_PATHS}. "
        "Run the pipeline first: python3 -m nas_rl.train"
    )


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Load image bytes, resize to 32x32, normalise, return NCHW float32."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0      # HWC [0,1]
    arr = arr.transpose(2, 0, 1)                        # CHW
    arr = (arr - MEAN) / STD                            # normalise
    return arr[np.newaxis, ...]                         # NCHW batch=1


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def run_inference(image_bytes: bytes) -> dict:
    """
    Run ONNX inference on raw image bytes.

    Returns dict with:
        top_class, top_confidence, top5 (list of {label, confidence}), latency_ms
    """
    session = _get_session()
    input_name = session.get_inputs()[0].name

    tensor = preprocess(image_bytes)

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]          # shape (10,)
    probs  = softmax(logits)

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [
        {'label': CIFAR10_CLASSES[i], 'confidence': float(probs[i])}
        for i in top5_idx
    ]

    return {
        'top_class':      CIFAR10_CLASSES[top5_idx[0]],
        'top_confidence': float(probs[top5_idx[0]]),
        'top5':           top5,
        'latency_ms':     round(latency_ms, 2),
    }
