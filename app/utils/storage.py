import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_bytes, convert_from_path

BASE_DIR = Path(__file__).resolve().parent.parent  # raíz del proyecto
DEFAULT_STORAGE_DIR = BASE_DIR / "storage"
MAX_BYTES = 500_000  # 0.5 MB


def get_storage_dir() -> Path:
    """
    Devuelve el directorio donde se guardarán las imágenes válidas.
    """
    env_dir = os.getenv("INE_STORAGE_DIR")
    if env_dir:
        d = Path(env_dir)
    else:
        d = DEFAULT_STORAGE_DIR

    d.mkdir(parents=True, exist_ok=True)
    return d


def build_storage_filename(
    request_id: str,
    original_filename: str,
    prefix: str = "ine",
) -> Path:
    """
    Construye un nombre de archivo tipo:
    ine_2025-12-04_153012_<request_id>_original.ext
    """
    storage_dir = get_storage_dir()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # extraer extensión, si existe
    orig_name = Path(original_filename).name
    return storage_dir / f"{prefix}_{ts}_{request_id}_{orig_name}"


def _compress_to_max_size(bgr: np.ndarray, max_bytes: int = MAX_BYTES) -> bytes:
    """
    Comprime una imagen BGR a JPEG reduciendo calidad progresivamente
    hasta que sea <= max_bytes. Devuelve bytes finales.
    """
    # Intentos progresivos de calidad JPEG (de mejor a peor)
    quality_steps = [95, 85, 75, 65, 55, 45, 35, 25]

    for q in quality_steps:
        encode_ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not encode_ok:
            continue
        jpg_bytes = buffer.tobytes()
        if len(jpg_bytes) <= max_bytes:
            return jpg_bytes

    # Si ninguna calidad logró quedar debajo de max_bytes,
    # devolvemos el último intento (el más comprimido).
    return jpg_bytes


def save_valid_image(
    request_id: str,
    filename: str,
    image: str,
    out_dir: str = "storage",
    max_bytes: int = MAX_BYTES,
) -> str:
    """
    Guarda la imagen enviada por la API asegurando que:
    - No exceda max_bytes (0.5MB por default)
    - Se almacene siempre como JPEG
    - Use request_id como prefijo para trazabilidad

    Devuelve el path final guardado.
    """

    os.makedirs(out_dir, exist_ok=True)

    ext = Path(filename).suffix.lower()
    is_pdf = ext == ".pdf"

    # 1) Convertir a BGR (imagen) según sea PDF o imagen normal

    if is_pdf:
        # Rasterizar primera página del PDF
        try:
            pages = convert_from_path(image, dpi=200)
        except Exception as e:
            raise ValueError(f"No se pudo convertir el PDF a imagen: {e}")

        if not pages:
            raise ValueError("El PDF no contiene páginas válidas.")
        print("Convert Success")
        pil_img = pages[0].convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        base_stem = Path(filename).stem or "pdf"
    else:
        image = Path(image).read_bytes()
        # Imagen (jpg/png/etc.)
        npimg = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("No se pudo decodificar la imagen enviada.")
        base_stem = Path(filename).stem or "img"

    # 2) Comprimir asegurando tamaño máximo
    final_bytes = _compress_to_max_size(bgr, max_bytes=max_bytes)
    # 3) Nombre final: siempre .jpg
    out_name = f"{request_id}_{base_stem}.jpg"
    out_path = Path(out_dir) / out_name

    with open(out_path, "wb") as f:
        f.write(final_bytes)

    return str(out_path)