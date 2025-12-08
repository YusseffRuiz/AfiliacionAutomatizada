import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import shutil
import time
import datetime

# IMPORTA tu lógica existente
from .image_processor import IDImageProcessor
from .id_parser import INEParser
from .helper import process_with_yolo_v2  # donde tengas esta función
from .ocr_agent import MistralOCRAgent, PaddleOCREngine
from .utils import health, storage

# ----------------- Modelos Pydantic de respuesta -----------------
class ErrorContext(BaseModel):
    ocr_engine: Optional[str] = None
    attempt: Optional[str] = None
    filename: Optional[str] = None
    stage: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class ErrorPayload(BaseModel):
    type: str                  # p.ej. "validation_error", "ocr_error", "model_error", image_error
    message: str               # mensaje entendible
    detail: Optional[str] = None  # detalle técnico más específico
    context: Optional[ErrorContext] = None

class INEApiError(Exception):
    def __init__(
        self,
        *,
        type: str,
        message: str,
        detail: Optional[str] = None,
        context: Optional[dict] = None,
        status_code: int = 400,
    ):
        self.type = type
        self.message = message
        self.detail = detail
        self.context = context or {}
        self.status_code = status_code
        super().__init__(message)

class INEData(BaseModel):
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None
    nombres: Optional[str] = None
    direccion: Optional[str] = None  # viene de "domicilio"
    curp: Optional[str] = None  # Validacion con api de curp
    fecha_nacimiento: Optional[str] = None  # formato ISO: YYYY-MM-DD // Por verse
    curp_validada : Optional[bool] = None  # validacion de curp con gob


class INEMeta(BaseModel):
    request_id: Optional[str] = None
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
agent_paddle = PaddleOCREngine(lang="es")
if not api_key:
    raise ValueError("Please set the MISTRAL_API_KEY environment variable.")
agent_mistral = MistralOCRAgent(api_key=api_key)


# -----------------Error Handling ---------------------
logger = logging.getLogger("ine_api")

@app.exception_handler(INEApiError)
async def ine_api_error_handler(request: Request, exc: INEApiError):
    payload = ErrorPayload(
        type=exc.type,
        message=exc.message,
        detail=exc.detail,
        context=ErrorContext(**exc.context) if exc.context else None,
    )

    # Log estructurado
    logger.error(
        "INEApiError",
        extra={
            "error_type": exc.type,
            "message": exc.message,
            "detail": exc.detail,
            "context": exc.context,
            "path": str(request.url),
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": payload.dict()},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    # Aquí atrapamos lo que no controlamos
    logger.exception("Unhandled exception in INE API", extra={"path": str(request.url)})

    payload = ErrorPayload(
        type="internal_error",
        message="Ocurrió un error inesperado procesando la credencial.",
        detail=str(exc),
        context=ErrorContext(
            extra={"path": str(request.url)}
        ),
    )

    return JSONResponse(
        status_code=500,
        content={"error": payload.dict()},
    )

# ----------------- Endpoint principal -----------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    components = {
        "yolo": health.check_yolo(processor),
        "paddle_ocr": health.check_paddle(agent_paddle),
        "mistral_ocr": health.check_mistral(agent_mistral),
        "parser": health.check_parser(parser),
    }

    all_ok = all(c["ok"] for c in components.values())

    status = "ok" if all_ok else "degraded"

    payload = {
        "status": status,
        "components": components,
    }

    status_code = 200 if all_ok else 503
    return JSONResponse(content=payload, status_code=status_code)

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
    request_id = str(uuid.uuid4().hex)[:4]

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
        if ocr_engine == "paddle": ## If added an uknown agent, use last resort ocr enginer = Tesseract
            agent = agent_paddle
        elif ocr_engine == "mistral":
            agent = agent_mistral
        else:
            agent = None

        # 3) Validacion de imagen
        img_bgr = processor.public_load_image(str(tmp_path), page=page)
        valid_img = health.validate_image_quality(img_bgr, filename=file.filename)

        if not valid_img:
            raise INEApiError(
                type=valid_img["type"],
                message=valid_img["message"],
                detail=valid_img["detail"],
                context=valid_img["context"],
                status_code=valid_img["status_code"],
            )

        # 4) Ejecutar pipeline con candidatos de YOLO + parser, regresa el Dict
        result = process_with_yolo_v2(processor=processor, parser=parser, agent=agent, ine_imagen=img_bgr)

        score = int(result.get("score", 0))
        if score == 0:
            raise INEApiError(
                type="ocr_error",
                message="No se pudo extraer texto legible de la credencial.",
                detail="El motor OCR devolvió texto vacío o solo ruido.",
                context={
                    "ocr_engine": ocr_engine,
                    "filename": file.filename,
                    "stage": "ocr",
                },
                status_code=422,
            )

        ## 4.5) Guardar imagen en disco para futuros entrenamientos.
        try:
            raw_bytes = Path(tmp_path).read_bytes()
            storage.save_valid_image(
                request_id=request_id,
                original_filename=file.filename or "upload",
                image=raw_bytes,
            )
        except Exception as e:
            # No queremos que falle toda la API solo porque no se pudo guardar
            logger.warning(
                "No se pudo guardar imagen válida",
                extra={
                    "request_id": request_id,
                    "filename": file.filename,
                    "error": str(e),
                },
            )

        # 5) Mapear a lo que necesita el CRM

        data = INEData(
            apellido_paterno=result.get("apellido_paterno"),
            apellido_materno=result.get("apellido_materno"),
            nombres=result.get("nombres"),
            direccion=result.get("domicilio"),
            curp=result.get("curp"),
            fecha_nacimiento=normalize_fecha_ddmmyyyy_to_iso(result.get("fecha_nacimiento")),
            curp_validada=result.get("validated_curp"),
        )

        meta = INEMeta(
            request_id=request_id,
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
