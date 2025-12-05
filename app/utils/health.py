import os
from typing import Dict, Any
import cv2
import numpy as np

def validate_image_quality(
    img: np.ndarray,
    filename: str,
    min_side: int = 600,
    max_side: int = 5000,
    min_contrast: float = 12.0,
    min_edges: int = 500
):
    """
    Valida que la imagen sea legible antes de procesarla por OCR.
    Revisa:
    - corrupción
    - tamaño mínimo/máximo
    - no ser totalmente negra/blanca
    - contraste mínimo
    - bordes suficientes (indica estructura del documento)
    """

    # --- 1) Verificar que la imagen está cargada correctamente ---
    if img is None or img.size == 0:
        return {
            "type":"invalid_image",
            "message":"La imagen no se pudo decodificar o está corrupta.",
            "detail":"OpenCV devolvió una matriz vacía.",
            "context":{"filename": filename, "stage": "image_validation"},
            'status_code':400,
        }

    h, w = img.shape[:2]

    # --- 2) Checar tamaño mínimo/máximo ---
    if min(h, w) < min_side:
        return {
            "type": "invalid_image",
            "message": "La imagen es demasiado pequeña para procesarse.",
            "detail": "Dimensiones: {w}×{h}, mínimo requerido: {min_side}px.",
            "context": {"filename": filename, "stage": "image_validation"},
            'status_code': 400,
        }

    if max(h, w) > max_side:
        return {
            "type": "invalid_image",
            "message": "La imagen es demasiado grande para procesarse.",
            "detail": f"Dimensiones: {w}×{h}, mínimo requerido: {min_side}px.",
            "context": {"filename": filename, "stage": "image_validation"},
            'status_code': 400,
        }

    # Convertimos a gris una sola vez para análisis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 3) Detectar si es completamente blanca o negra ---
    mean_val = gray.mean()
    if mean_val < 10:
        return {
            "type": "invalid_image",
            "message": "La imagen está demasiado oscurecida.",
            "detail": f"Media de intensidad: {mean_val}",
            "context": {"filename": filename, "stage": "image_validation"},
            'status_code': 400,
        }

    if mean_val > 245:
        return {
            "type": "invalid_image",
            "message": "La imagen está demasiado iluminada.",
            "detail": f"Media de intensidad: {mean_val}",
            "context": {"filename": filename, "stage": "image_validation"},
            'status_code': 400,
        }

    # --- 4) Medir contraste (std de intensidades) ---
    std_val = gray.std()
    if std_val < min_contrast:
        return {
            "type": "invalid_image",
            "message": "La imagen tiene muy bajo contraste y no es legible.",
            "detail": f"Contraste (std): {std_val}, mínimo requerido: {min_contrast}.",
            "context": {"filename": filename, "stage": "image_validation"},
            'status_code': 400,
        }

    # --- 5) TODO: Detectar bordes para validar estructura, identificar rectangulos. ---

    # If everything is good:
    return True

def check_yolo(processor) -> Dict[str, Any]:
    try:
        ok = hasattr(processor, "model") and processor.model is not None
        return {"ok": bool(ok), "detail": "loaded" if ok else "model is None"}
    except Exception as e:
        return {"ok": False, "detail": f"exception: {e}"}


def check_paddle(paddle_engine) -> Dict[str, Any]:
    try:
        # PaddleOCREngine tiene atributo .ocr si está inicializado
        ok = hasattr(paddle_engine, "ocr") and paddle_engine.ocr is not None
        return {"ok": bool(ok), "detail": "initialized" if ok else "ocr is None"}
    except Exception as e:
        return {"ok": False, "detail": f"exception: {e}"}


def check_mistral(mistral_engine) -> Dict[str, Any]:
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return {"ok": False, "detail": "MISTRAL_API_KEY not set"}

        if mistral_engine is None:
            return {"ok": False, "detail": "mistral engine is None"}

        # Versión ligera: solo revisar que el cliente está construido
        # Si luego quieres, puedes agregar un ping real con timeout corto.
        return {"ok": True, "detail": "initialized"}
    except Exception as e:
        return {"ok": False, "detail": f"exception: {e}"}


def check_parser(parser) -> Dict[str, Any]:
    try:
        ok = parser is not None
        return {"ok": bool(ok), "detail": "initialized" if ok else "parser is None"}
    except Exception as e:
        return {"ok": False, "detail": f"exception: {e}"}
