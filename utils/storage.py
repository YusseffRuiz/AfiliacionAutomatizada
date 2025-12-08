import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # raíz del proyecto
DEFAULT_STORAGE_DIR = BASE_DIR / "storage"


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


def save_valid_image(
    image: bytes,
    request_id: str,
    original_filename: str,
    prefix: str = "ine",
) -> Path:
    """
    Guarda el contenido binario de la imagen/PDF válida en disco.
    Devuelve la ruta donde se guardó.
    """
    out_path = build_storage_filename(
        request_id=request_id,
        original_filename=original_filename,
        prefix=prefix,
    )
    with open(out_path, "wb") as f:
        f.write(image)
    return out_path
