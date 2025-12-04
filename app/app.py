import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import tempfile
import shutil
import time
import datetime

# IMPORTA tu lógica existente
from .image_processor import IDImageProcessor
from .id_parser import INEParser
from .helper import process_with_yolo_candidates_tesseract, process_with_yolo_candidates_mistral, process_with_yolo_candidates_paddle  # donde tengas esta función
from .ocr_agent import SimpleOCRAgent, PaddleOCREngine


# ----------------- Modelos Pydantic de respuesta -----------------


class INEData(BaseModel):
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None
    nombres: Optional[str] = None
    direccion: Optional[str] = None  # viene de "domicilio"
    curp: Optional[str] = None  # Validacion con api de curp
    fecha_nacimiento: Optional[str] = None  # formato ISO: YYYY-MM-DD // Por verse


class INEMeta(BaseModel):
    score: int
    parser_version: str
    processing_ms: int
    warnings: List[str] = []


class INEOKResponse(BaseModel):
    status: str = "ok"
    data: INEData
    meta: INEMeta


class INEErrorDetail(BaseModel):
    type: str
    message: str
    suggestion: Optional[str] = None


class INEErrorResponse(BaseModel):
    status: str = "error"
    error: INEErrorDetail


# ----------------- Helpers -----------------


def normalize_fecha_ddmmyyyy_to_iso(fecha: Optional[str]) -> Optional[str]:
    """
    Convierte 'DD/MM/YYYY' -> 'YYYY-MM-DD'.
    Si no se puede parsear, regresa None.
    Por revisarse.
    """
    if not fecha:
        return None
    try:
        dt = datetime.datetime.strptime(fecha, "%d/%m/%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def build_warnings(result: dict) -> List[str]:
    warnings = []
    if not result.get("curp"):
        warnings.append("curp_no_detectada")
    if not result.get("nombre_completo"):
        warnings.append("nombre_no_detectado")
    if not result.get("domicilio"):
        warnings.append("domicilio_no_detectado")
    if not result.get("fecha_nacimiento"):
        warnings.append("fecha_nacimiento_no_detectada")
    return warnings

def ine_pipeline(processor, parser, ine_imagen, page=0, ocr_engine="paddle"):
    if ocr_engine == "paddle":
        agent = PaddleOCREngine(lang="es")

        result = process_with_yolo_candidates_paddle(processor=processor, paddle_agent=agent, parser=parser,
                                                     ine_imagen=ine_imagen)
        return result
    elif ocr_engine == "tesseract":
        result = process_with_yolo_candidates_tesseract(processor=processor, parser=parser, ine_imagen=ine_imagen, page=page)
        return result
    elif ocr_engine == "mistral":
        if not api_key:
            raise ValueError("Please set the MISTRAL_API_KEY environment variable.")
        agent = SimpleOCRAgent(api_key=api_key)
        result = process_with_yolo_candidates_mistral(processor=processor, mistral_agent=agent, parser=parser,
                                                 ine_imagen=str(ine_imagen), page=page)
        return result
    else:
        raise ValueError("OCR engine no reconocido")


# ----------------- Inicializar FastAPI y tus objetos -----------------


app = FastAPI(
    title="INE OCR API",
    description="Servicio para extraer datos de credenciales INE",
    version="1.0.0",
)

processor = IDImageProcessor(
    yolo_model_path="models/YOLOV8_INE_V2.pt",  # ajusta al modelo que estés usando
    conf_threshold=0.4,
    debug_dir="debug_dir",
    save_debug_images=False,
)
parser = INEParser()
BASE_DIR = Path(__file__).resolve().parent.parent   # sube dos carpetas
ENV_PATH = BASE_DIR / "tokens.env"
load_dotenv(ENV_PATH)
api_key = os.getenv("MISTRAL_API_KEY")



# ----------------- Endpoint principal -----------------


@app.post(
    "/api/ine/parse",
    response_model=INEOKResponse,
    responses={
        400: {"model": INEErrorResponse},
        415: {"model": INEErrorResponse},
        422: {"model": INEErrorResponse},
        500: {"model": INEErrorResponse},
    },
)
async def parse_ine(
    file: UploadFile = File(...),
    source: Optional[str] = Form(None),
    return_debug: bool = Form(False),
    page: int = Form(0),
    ocr_engine: str = "paddle",
):
    start = time.time()
    tmp_path: Optional[Path] = None

    # 1) Validar tipo de archivo
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/tiff",
        "application/pdf",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail={
                "type": "unsupported_media_type",
                "message": f"Formato no soportado: {file.content_type}. Use JPG, PNG, TIFF o PDF.",
            },
        )

    try:
        # 2) Guardar archivo temporalmente
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)


        # 4) Ejecutar pipeline con candidatos de YOLO + parser
        result = ine_pipeline(processor=processor, parser=parser, ine_imagen=tmp_path, page=page, ocr_engine=ocr_engine)

        score = int(result.get("score", 0))

        # 5) Mapear a lo que necesita el CRM

        data = INEData(
            apellido_paterno=result.get("apellido_paterno"),
            apellido_materno=result.get("apellido_materno"),
            nombres=result.get("nombres"),
            direccion=result.get("domicilio"),
            curp=result.get("curp"),
            fecha_nacimiento=normalize_fecha_ddmmyyyy_to_iso(result.get("fecha_nacimiento")),
        )

        meta = INEMeta(
            score=score,
            parser_version="ine-mvp-v1",
            processing_ms=int((time.time() - start) * 1000),
            warnings=build_warnings(result),
        )

        response = INEOKResponse(status="ok", data=data, meta=meta)
        return JSONResponse(content=response.model_dump())

    except RuntimeError as e:
        # Errores de negocio tipo "no id detectada", etc.
        err = INEErrorResponse(
            status="error",
            error=INEErrorDetail(
                type="no_id_detected",
                message=str(e),
                suggestion="Asegúrese de que la credencial completa sea visible, con buena iluminación.",
            ),
        )
        raise HTTPException(status_code=422, detail=err.dict()["error"])

    except HTTPException:
        # Re-lanzar HTTPExceptions tal cual
        raise

    except Exception as e:
        # Log interno recomendado aquí
        err = INEErrorResponse(
            status="error",
            error=INEErrorDetail(
                type="internal_error",
                message="Ocurrió un error inesperado procesando la credencial.",
            ),
        )
        raise HTTPException(status_code=500, detail=err.dict()["error"])

    finally:
        # 6) Borrar archivo temporal
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
