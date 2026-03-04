import logging
import os
import sys
import uuid


from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import shutil
import time
import datetime
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.status import HTTP_403_FORBIDDEN

from .image_processor import IDImageProcessor
from .id_parser import INEParser
from .helper import process_with_yolo_v2  # donde tengas esta función
from .ocr_agent import MistralOCRAgent, PaddleOCREngine
from .utils import health, storage

# ----------------- Modelos Pydantic de respuesta -----------------
class ErrorContext(BaseModel):
    model_config = ConfigDict(extra='allow')
    ocr_engine: Optional[str] = None
    attempt: Optional[str] = None
    filename: Optional[str] = None
    stage: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class ErrorPayload(BaseModel):
    type: str                  # p.ej. "validation_error", "ocr_error", "model_error", image_error
    message: str               # mensaje entendible
    detail: Optional[str] = None  # detalle técnico más específico
    context: Optional[ErrorContext] = None
    timestamp: Optional[str] = None

class INEApiError(Exception):
    def __init__(
        self,
        *,
        type: str,
        message: str,
        detail: Optional[str] = None,
        context: Optional[dict] = None,
        timestamp: Optional[str] = None,
        status_code: int = 400,
    ):
        self.type = type
        self.message = message
        self.detail = detail
        self.context = context or {}
        self.status_code = status_code
        self.timestamp = timestamp
        super().__init__(message)

class INEData(BaseModel):
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None
    nombres: Optional[str] = None
    sexo: Optional[str] = None
    direccion: Optional[str] = None  # viene de "domicilio"
    calle: Optional[str] = None
    colonia: Optional[str] = None
    municipio: Optional[str] = None
    ciudad : Optional[str] = None
    estado: Optional[str] = None
    codigo_postal: Optional[str] = None
    curp: Optional[str] = None  # Validacion con api de curp
    clave_elector: Optional[str] = None
    fecha_nacimiento: Optional[str] = None  # formato ISO: YYYY-MM-DD // Por verse
    curp_validada : Optional[bool] = None  # validacion de curp con gob
    vigencia: Optional[str] = None
    seccion: Optional[str] = None


class INEMeta(BaseModel):
    request_id: Optional[str] = None
    score: int
    parser_version: str
    processing_ms: int
    warnings: List[str] = []
    ocr_engine: Optional[str] = None


class INEOKResponse(BaseModel):
    status: str = "ok"
    data: INEData
    meta: INEMeta


class INEErrorDetail(BaseModel):
    type: str
    message: str
    suggestion: Optional[str] = None
    timestamp: Optional[str] = None


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
    if not result.get("codigo_postal"):
        warnings.append("codigo_postal_no_detectado")
    return warnings


# ----------------- Inicializar FastAPI y tus objetos -----------------
# Rate limiter, para evitar un ataque multiple y se bloquee tras ciertos intentos
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="INE OCR API",
    description="Servicio para extraer datos de credenciales INE",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. Configuración de API Key
API_KEY_NAME = "MAIN-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)



origins_list = ["*"] # Es importante especificar las URL del origen, es decir, de sybi


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins_list,           # Permite el Access-Control-Allow-Origin
#     allow_credentials=True,
#     allow_methods=["*"],           # Permite todos los métodos (GET, POST, etc.)
#     allow_headers=["*"],           # Permite todos los headers
# )
#
# # Middleware para confiar en los headers que envía IIS
# app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="127.0.0.1")


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


API_KEYS = os.getenv("MAIN_API_KEYS", "").split(",")

async def validar_api_key(header_key: str = Security(api_key_header)):
    if header_key and header_key in API_KEYS:
        print("Found key!")
        return header_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Acceso denegado: API Key inválida o ausente"
    )


# -----------------Error Handling ---------------------
logger = logging.getLogger("ine_api")

logging.basicConfig(
    level=logging.INFO,  # ⬅️ IMPORTANTE
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
)

@app.exception_handler(INEApiError)
async def ine_api_error_handler(request: Request, exc: INEApiError):
    payload = ErrorPayload(
        type=exc.type,
        message=exc.message,
        detail=exc.detail,
        context=ErrorContext(**exc.context) if exc.context else None,
        timestamp = exc.timestamp,
    )
    # Log estructurado
    logger.error(
        "INEApiError",
        extra={
            "error_type": exc.type,
            "error_message": exc.message,
            "error_detail": exc.detail,
            "error_context": exc.context,
            "path": str(request.url),
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": payload.model_dump(exclude_none=True)},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    # Aquí atrapamos lo que no controlamos
    logger.exception("Unhandled exception in INE API", extra={"path": str(request.url)})

    payload = ErrorPayload(
        type="internal_error",
        message="Ocurrió un error inesperado procesando la credencial.",
        detail=str(exc),
        timestamp=str(datetime.datetime.now()),
        context=ErrorContext(
            extra={"path": str(request.url)}
        ),
    )

    return JSONResponse(
        status_code=500,
        content={"error": payload.model_dump(exclude_none=True)},
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
# @limiter.limit("10/minute")  # Limite de 10 peticiones por minuto por IP
async def parse_ine(
    # request: Request,
    file: UploadFile = File(...),
    card_id: Optional[str] = "1",
    # token: str = Depends(validar_api_key),
    page: int = Form(0),
    ocr_engine: str = "mistral",
):
    start = time.time()
    tmp_path: Optional[Path] = None
    request_id = str(uuid.uuid4().hex)[:4]  # Caben 65,536 requests

    # 1) Validar tipo de archivo
    allowed_types = {
        "image/jpeg",
        "image/jpg",
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
                "timestamp": str(datetime.datetime.now()),
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
        valid_img = health.validate_image_quality(img_bgr, filename=file.filename, processor=processor)

        if not valid_img:
            raise INEApiError(
                type=valid_img["type"],
                message=valid_img["message"],
                detail=valid_img["detail"],
                context=valid_img["context"],
                status_code=valid_img["status_code"],
                timestamp=str(datetime.datetime.now()),
            )
        # print("Valid image")
        # 4) Ejecutar pipeline con candidatos de YOLO + parser, regresa el Dict
        result = process_with_yolo_v2(processor=processor, parser=parser, agent=agent, ine_imagen=str(tmp_path))
        if 'error'in result:
            raise INEApiError(
                type="ocr_error",
                message="No se pudo extraer texto legible de la credencial.",
                detail=result['error'],
                context={
                    "ocr_engine": str(ocr_engine),
                    "filename": str(file.filename),
                    "stage": "ocr",
                },
                timestamp=str(datetime.datetime.now()),
                status_code=422,
            )
        score = int(result.get("score", 0))
        print("Done processing results", score)
        if score == 0:
            raise INEApiError(
                type="ocr_error",
                message="No se pudo extraer texto legible de la credencial.",
                detail="El motor OCR no encontró el campo de la vigencia",
                context={
                    "ocr_engine": str(ocr_engine),
                    "filename": str(file.filename),
                    "stage": "ocr",
                },
                timestamp=str(datetime.datetime.now()),
                status_code=422,
            )
        # print("Done scoring")
        valid_ine_expiracy = result.get("vigencia", "0000")
        print("INE expiracy: ", valid_ine_expiracy)
        if valid_ine_expiracy == "0000" or valid_ine_expiracy is None:
            raise INEApiError(type="Error en Vigencia",
                message="No se pudo extraer la vigencia de la INE, volver a tomar la fotografia",
                detail="El motor OCR devolvió texto vacío o solo ruido.",
                context={
                    "ocr_engine": str(ocr_engine),
                    "filename": str(file.filename),
                    "stage": "Verificacion",
                },
                timestamp=str(datetime.datetime.now()),
                status_code=422,
            )
        else:
            try:
                valid_ine_expiracy = int(valid_ine_expiracy[-4:]) # Obtenemos los ultimos 4 digitos
            except ValueError:
                raise INEApiError(type="Error en Vigencia",
                                  message="No se pudo extraer la vigencia de la INE, volver a tomar la fotografia",
                                  detail="El motor OCR devolvió texto vacío o solo ruido.",
                                  context={
                                      "ocr_engine": str(ocr_engine),
                                      "filename": str(file.filename),
                                      "stage": "Verificacion",
                                  },
                                  timestamp=str(datetime.datetime.now()),
                                  status_code=422,
                                  )
            if valid_ine_expiracy < int(datetime.date.today().year):
                raise INEApiError(type="Error en Vigencia",
                                  message="INE caduca",
                                  detail="Se encontró que la vigencia es menor al año actual",
                                  context={
                                      "ocr_engine": str(ocr_engine),
                                      "filename": str(file.filename),
                                      "stage": "Verificacion",
                                  },
                                  timestamp=str(datetime.datetime.now()),
                                  status_code=422,
                                  )
        ## 4.5) Guardar imagen en disco para futuros entrenamientos.
        try:
            storage.save_valid_image(  # Guardado de la imagen en storage
                request_id=request_id,
                filename=file.filename or "upload",
                image=str(tmp_path),
                card_id=card_id,
                user_name=result.get("nombre_completo"),
            )
            print("Correct storing")
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
            sexo=result.get("sexo"),
            direccion=result.get("domicilio"),
            codigo_postal=result.get("codigo_postal"),
            calle=result.get("calle"),
            colonia=result.get("colonia"),
            ciudad=result.get("ciudad"),
            municipio=result.get("municipio"),
            estado=result.get("estado"),
            curp=result.get("curp"),
            fecha_nacimiento=normalize_fecha_ddmmyyyy_to_iso(result.get("fecha_nacimiento")),
            curp_validada=result.get("validated_curp"),
            clave_elector=result.get("clave_elector"),
            seccion=result.get("seccion"),
            vigencia=result.get("vigencia")
        )

        meta = INEMeta(
            request_id=request_id,
            score=score,
            ocr_engine=ocr_engine,
            parser_version="ine-mvp-v1",
            processing_ms=int((time.time() - start) * 1000),
            warnings=build_warnings(result),
        )

        response = INEOKResponse(status="ok", data=data, meta=meta)
        return JSONResponse(content=response.model_dump(exclude_none=True))
    except INEApiError:
        raise

    except RuntimeError as e:
        # Errores de negocio tipo "no id detectada", etc.
        err = INEErrorResponse(
            status="error",
            error=INEErrorDetail(
                type="no_id_detected",
                message=str(e),
                suggestion="Asegúrese de que la credencial completa sea visible, con buena iluminación.",
                timestamp=str(datetime.datetime.now()),
            ),
        )
        raise HTTPException(status_code=422, detail=err.model_dump(exclude_none=True)["error"])

    except HTTPException:
        # Re-lanzar HTTPExceptions
        raise

    except Exception as e:
        err = INEErrorResponse(
            status="error",
            error=INEErrorDetail(
                type="internal_error",
                message=f"Ocurrió un error inesperado procesando la credencial: {e}",
                timestamp=str(datetime.datetime.now()),
            ),
        )
        raise HTTPException(status_code=500, detail=err.model_dump(exclude_none=True)["error"])

    finally:
        # 6) Borrar archivo temporal
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
