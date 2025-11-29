FROM python:3.11-slim

LABEL authors="AdanYDR"

ENV DEBIAN_FRONTEND=noninteractive


# 3) Instalar dependencias de sistema (Tesseract + libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-spa \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 4) Crear directorio de la app
WORKDIR /app

# 5) Copiar requirements e instalar
COPY requirements.txt /app/requirements.txt

RUN  pip install --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --default-timeout=120 --no-cache-dir -r requirements.txt

# 6) Copiar el resto del código
COPY . /app

# 7) Exponer puerto para uvicorn
EXPOSE 8000

# 8) Comando por defecto (levantar FastAPI)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]