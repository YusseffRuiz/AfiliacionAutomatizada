# INE OCR – Sistema Automático de Afiliación por Lectura de Credenciales INE

Pipeline completo de detección, corrección de perspectiva, OCR (Paddle / Mistral / Tesseract) y extracción estructurada de datos.

## 📌 Descripción General

Este proyecto implementa un sistema robusto de lectura y extracción de información desde credenciales del INE (Instituto Nacional Electoral – México) utilizando visión por computadora y OCR avanzado.

Permite:

Detectar la credencial INE dentro de una imagen usando YOLOv8 (modelo entrenado específicamente).

Corregir perspectiva y normalizar la imagen con OpenCV para obtener imagen limpia de INE.

Leer texto usando distintos motores OCR:

- PaddleOCR (default)

- Mistral OCR (API)

- Tesseract OCR (Last Resort)

Extraer datos estructurados:

- Apellido paterno

- Apellido materno

- Nombres

- Nombre completo

- CURP

- Clave de elector

- Sexo

- Domicilio

- Fecha de nacimiento

- Vigencia

Todo expuesto mediante una API FastAPI para integración con sistemas CRM o aplicaciones móviles.

## 🚀 Características principales
✔ Detección robusta de credencial

Mediante un modelo YOLOv8 entrenado con >380 imágenes etiquetadas.

✔ Corrección automática de perspectiva

Warping mediante homografía para mejorar el OCR.

✔ OCR multi-engine

Seleccionable por request:

- PaddleOCR (rápido, muy preciso)

- Mistral OCR (vía API, alta calidad)

- Tesseract (fallback)

✔ Limpieza de texto y parsing especializado para INE

Regex avanzados para formatos de CURP, fechas, secciones y heurísticas de nombres.

✔ Failover robusto

Si una detección falla:

- se prueban múltiples bounding boxes,

- se reintenta OCR con fallback,

- se retorna el mejor resultado posible.

✔ API lista para producción

- Manejo de errores

- Compatibilidad con contenedores Docker

## 📁 Estructura del Proyecto
    app/
    │── app.py                 # FastAPI main app
    │── image_processor.py     # YOLO detection + warp + preprocessing
    │── ocr_engines.py         # Paddle, Mistral, Tesseract OCR wrappers
    │── ine_parser.py          # Parsing estructurado del INE
    │── utils/                 # utilidades varias
    models/
    │── YOLOV8_INE_V2.pt       # Modelo YOLO entrenado
    tokens.env                 # Llave API Mistral
    README.md

🔧 Instalación

2️⃣ Instalar dependencias

    pip install -r requirements.txt

3️⃣ Configurar variables de entorno

Crea tokens.env:

    MISTRAL_API_KEY=tu_api_key


📤 Uso del endpoint principal

    POST /api/ine/parse

Parámetros:

    Campo	        | Tipo                     |	Descripción
    file	        | UploadFile               |	Imagen/PDF de INE
    ocr_engine      | paddle/mistral/tesseract |	Motor OCR (default: paddle)
    page	        | int	                   |    Página del PDF
    source	        | str	                   |    Opcional
    return_debug	| bool	                   |    Retorna imágenes intermedias
Ejemplo con curl:

    curl -X POST "http://localhost:8000/api/ine/parse" \
        -F "file=@INE_13.jpg" \
        -F "ocr_engine=paddle"

🧪 Ejemplo de Respuesta

    {
      "apellido_paterno": "LOPEZ",
      "apellido_materno": "HERNANDEZ",
      "nombres": "ANA ISABEL",
      "nombre_completo": "LOPEZ HERNANDEZ ANA ISABEL",
      "sexo": "MUJER",
      "domicilio": "AV JAIME TORRES BODET 2963 A22, COL EL SAUZ 45608, SAN PEDRO TLAQUEPAQUE JAL",
      "clave_elector": "LPHRAN72010314M702",
      "curp": "LOHA720103MJCPRN01",
      "fecha_nacimiento": "03/01/1972",
      "vigencia": "2023-2033"
    }

## Errores de la API
    200 → todo OK, datos útiles.
    
    400 → problema con la imagen (corrupta, mal subida, formato no soportado).
    
    422 → OCR/parseo no logró campos mínimos (pero la imagen era válida).
    
    500 → error interno inesperado.

## 🧩 Roadmap

- Mejorar dataset y entrenamiento YOLO (v3). 
  - Entrenar con >300 INE en el mismo angulo para identificar campos y letras.
  - Modelo de segmentación por zonas: NOMBRE, CURP, DOMICILIO
- App móvil integrada.
- Almacenamiento de imagenes en servidor.
- Optimización para ejecución sobre GPU.

## 🤝 Contribuciones

Contribuciones, mejoras y PRs son bienvenidos.
Puedes abrir issues para reportar casos difíciles o enviar nuevas muestras de INEs.

## 📄 Licencia y permisos de uso

Este software se proporciona para uso interno y desarrollo de soluciones de afiliación, análisis de documentos y automatización empresarial.
Se permite:

- Usar el código dentro de proyectos institucionales.

- Modificarlo según necesidades operativas.

- Integrarlo con otros sistemas del ecosistema tecnológico de la organización.

**Obligación de atribución**

Si este software se utiliza en:

- aplicaciones comerciales,

- módulos integrados en otros proyectos,

- publicaciones técnicas o científicas,

- presentaciones o demostraciones externas,

deberá otorgarse crédito explícito al autor y al proyecto original de la siguiente forma:

    Sistema de extracción automatizada de datos INE desarrollado por:
    Adán Domínguez – Innovación Tecnológica Medical Life

o, en formato informal:

    Basado en el módulo OCR/INE desarrollado por Adán Domínguez (Medical Life).



Si se requiere los pesos del modelo YOLO, favor de escribir al desarrollador.

