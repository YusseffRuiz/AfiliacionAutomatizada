import base64
import json

import cv2
import numpy as np
from mistralai import Mistral, ImageURLChunk
from paddleocr import PaddleOCR


class PaddleOCREngine:
    def __init__(self, lang="es"):
        # Paddle usa "es" como español + multipropósito
        self.ocr = PaddleOCR(use_doc_orientation_classify=False,
                             # Disables document orientation classification model via this parameter
                             use_doc_unwarping=False,  # Disables text image rectification model via this parameter
                             use_textline_orientation=False,
                             # Disables text line orientation classification model via this parameter
                             lang=lang
                             )  # We dont need any of those.
        self.max_size = 4000 # Max size to handle images for Paddle

    def run(self, img: np.ndarray) -> str:
        """
        Recibe una imagen como np.ndarray (BGR o RGB)
        y devuelve texto plano concatenado.
        """
        # Paddle OCR requiere RGB
        h, w = img.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Normalizar a RGB 3 canales
        if img.ndim == 2:
            # gris -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Formato de imagen no soportado para OCR: shape={img.shape}")



        results = self.ocr.ocr(img_rgb)
        lines = []
        for res in results:  # cada res es un dict como el que pegaste
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])

            for text, score in zip(rec_texts, rec_scores):
                # opcional: filtrar basura de score muy bajo
                if score < 0.5:
                    continue
                lines.append(text)

        return "\n".join(lines)


class SimpleOCRAgent:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)

    def process_document(self, document_url):
        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": document_url
            },
            include_image_base64=True
        )
        return response

    def process_local_image(self, img):
        # Open and read the image file in binary mode
        success, encoded_img = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError("No se pudo codificar la imagen.")

        # Convertimos los bytes a base64
        encoded_bytes = encoded_img.tobytes()

        # Convert binary data to a base64 encoded string
        encoded_image = base64.b64encode(encoded_bytes).decode('utf-8')
        base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document=ImageURLChunk(image_url=base64_data_url),
        )
        response_dict = json.loads(response.model_dump_json())
        raw_text = self.mistral_ocr_to_raw_text(response_dict)

        return raw_text

    def mistral_ocr_to_raw_text(self, mistral_response) -> str:
        """
        Toma la respuesta del OCR de Mistral (dict) y devuelve texto plano
        listo para alimentar a INEParser.parse().
        """
        pages = mistral_response.get("pages", [])
        all_lines = []

        for page in pages:
            md = page.get("markdown", "") or ""

            # # 1) Quitar líneas de imágenes tipo: ![img-0.jpeg](img-0.jpeg)
            # md = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md)

            # 2) Separar en líneas y limpiar encabezados Markdown (#, ##, etc.)
            for line in md.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Quitar prefijos de título markdown (#, ##, ###, etc.)
                # line = re.sub(r"^#+\s*", "", line).strip()
                if line:
                    all_lines.append(line)

        raw_text = "\n".join(all_lines)
        return raw_text