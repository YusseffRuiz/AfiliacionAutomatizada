# Sistema de Afiliación Automática mediante OCR de INE

## 🪪 Descripción general

Este proyecto implementa un sistema completo para la extracción automática de datos de credenciales INE a partir de imágenes o PDFs.
El pipeline utiliza:

YOLOv8 (modelo entrenado) para detectar la tarjeta INE en la imagen.

Corrección de perspectiva para normalizar la tarjeta.

Preprocesamiento avanzado de OCR (denoise, contraste, escalado).

Tesseract OCR para extraer texto.

Parser especializado que estructura la información en campos como:

apellido paterno

apellido materno

nombres

dirección

CURP

fecha de nacimiento

vigencia

sexo

El sistema fue diseñado para integrarse a un CRM que realiza afiliación de clientes, reduciendo tiempo de captura y errores manuales.

## Arquitectura del Sistema

           📤 Imagen/PDF (INE)
                   │
                   ▼
       1. YOLOv8: detección de credencial
                   │ (múltiples candidatos)
                   ▼
     2. Warp + normalización de perspectiva
                   │
                   ▼
        3. Preprocesamiento OCR:
           - Denoise avanzado
           - CLAHE (contraste local)
           - Super-resize (escalado)
                   │
                   ▼
           4. Tesseract OCR
                   │
                   ▼
         5. Parser especializado INE
                   │
                   ▼
     6. Selección del "mejor resultado"
         (scoring de campos detectados)
                   │
                   ▼
           📦 JSON estructurado

## Requerimientos
* Python 3.11
* Tesseract OCR (con idioma español)
* Poppler (solo si se procesan PDFs)
* Bibliotecas Python:
* ultralytics
* opencv-python
* numpy
* pdf2image
* pytesseract
* fastapi
* uvicorn
* pydantic
* pillow


## Flujo interno del procesamiento
1. Detección de INE con YOLOv8

El sistema ejecuta YOLO sobre la imagen completa y obtiene todas las detecciones posibles (candidatos).
Para cada candidato:

* recorta el bounding box,

* aplica un pequeño margen,

* ejecuta un warp de perspectiva,

* prepara la imagen para OCR.

Esto permite fallback automático si la detección principal fue incorrecta.

2. Preprocesamiento para OCR

Se aplica:

* Conversión a grises

* Denoising adaptado

* CLAHE (aumento de contraste local)

* Upscaling 3× con interpolación cúbica

El objetivo es maximizar legibilidad de texto antes de mandarlo a Tesseract.

3. OCR con Tesseract
Se usa configuración optimizada. Y se ajusta dinámicamente por zonas más delicadas (nombre, CURP, clave).

```    
--oem 3      # LSTM (OCR moderno)

--psm 6      # Bloques de texto semi-estructurados
```


4. Parser de campos

El parser:

* normaliza el texto

* limpia ruido y caracteres basura

Encuentra secciones:

- Nombre Completo (Apellido Paterno / Materno / Nombre(s))
- Sexo
- Domicilio (CALLE / AV / COL / CP)
- Código Postal
- CURP (Con Validación)
- Clave de Elector
- Fecha de nacimiento
- Vigencia
- Sección


Extrae nombre completo usando heurísticas robustas:

- ignora tokens muy cortos,
- permite nombres compuestos,
- tolera errores de OCR.

También infiere:

- sexo y fecha de nacimiento desde la CURP (si OCR falla).

Finalmente  el parser genera una salida en formato Json:

```
{
  "apellido_paterno": "...",
  "apellido_materno": "...",
  "nombres": "...",
  "nombre_completo": "...",
  "domicilio": "...",
  "curp": "...",
  "clave_elector": "...",
  "fecha_nacimiento": "YYYY-MM-DD",
  "vigencia": "YYYY-YYYY"
}
```

## Entrega de la API
Se entrega el siguiente contrato con la llamada de la API, que se pueden modificar directamente de los datos extraídos del parser.

```
 data = INEData(
            apellido_paterno,
            apellido_materno,
            nombres,
            sexo,
            direccion,
            codigo_postal,
            curp,
            fecha_nacimiento,
            curp_validada,
            clave_elector,
            seccion=,
            vigencia=
        )

        meta = INEMeta(
            request_id,
            score,
            parser_version=,
            processing_ms,
            warnings,
        )
```

## Selección del mejor candidato
Si YOLO produjo varias detecciones:

* cada recorte se procesa,
* cada resultado se “puntúa” según:
* número de campos válidos,
* calidad del texto,
* presencia de CURP / clave elector,
* coherencia de nombre y domicilio.

Se elige el resultado con mayor score.

## Cómo ejecutar localmente

Ejecutar la API

```
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000 
```
Hacer una prueba con curl:
```
curl -X POST "http://127.0.0.1:8000/api/ine/parse" \
  -F "file=@imagenes_prueba/IneAdan.pdf"
```


## Trabajo a Futuro


Entrenar YOLOv8 multicampo para detectar zonas de:
* nombre
* CURP
* clave elector
* domicilio
* fecha de nacimiento

Implementar un modelo end-to-end tipo Donut (OCR transformer).