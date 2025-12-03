import base64
import json

import cv2
from mistralai import Mistral, ImageURLChunk


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