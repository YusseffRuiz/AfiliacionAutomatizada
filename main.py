from image_processor import IDImageProcessor
import matplotlib.pyplot as plt
import pytesseract
from id_parser import INEParser
import json
import cv2

YOLO_PATH = "models/YOLOV8_INE_V2.pt"


def score_parse_result(data_out: dict) -> int:
    """
    Asigna un score simple según cuántos campos clave se llenaron.
    Ajusta pesos según te convenga.
    """
    score = 0
    if data_out.get("curp"):
        score += 4
    if data_out.get("clave_elector"):
        score += 2
    if data_out.get("nombre_completo"):
        score += 2
    if data_out.get("domicilio"):
        score += 2
    if data_out.get("fecha_nacimiento"):
        score += 1
    if data_out.get("sexo"):
        score += 1
    return score

def process_with_yolo_candidates(
    processor: IDImageProcessor,
    parser: INEParser,
    ine_imagen: str,
    page: int = 0,
    max_candidates: int = 3,
    score_ok_threshold: int = 6,
) -> dict:
    """
    Prueba varios bounding boxes de YOLO (ordenados por confianza),
    se queda con el que produzca mejor parse. Si alguno supera
    cierto umbral de score, se detiene ahí.
    """
    best_data = None
    best_score = -1

    crops = processor.get_document_crops(ine_imagen, page=page, max_candidates=max_candidates)

    config = r"--psm 6 --oem 1 -c preserve_interword_spaces=1"
    config_name = r"--psm 6 --oem 1 -c preserve_interword_spaces=1"
    # PSM 6 = texto linea por linea en horizontal, oem = interpretador, 0 es automatico

    for i, crop in enumerate(crops):
        name_roi = processor.get_roi_name(crop, x1=30, y1=24, x2=42, y2=50, scale = 3.5, h_denoise= 20,
                                          search_window_size = 21, alpha_contrast = 1.8, beta_brightness = -10)
        pre = processor.public_preprocess_for_ocr(crop, scale=3.0, h = 26, searchwindowssize=21, clahe_clip_limit=3.6,
                                                alpha_contrast=1.8, beta_brightness=-26)

        texto = pytesseract.image_to_string(pre, lang="spa", config=config)
        texto_nombre = pytesseract.image_to_string(name_roi, lang="spa", config=config_name)

        name_out = parser.parse(texto_nombre)
        data_full = parser.parse(texto)   # tu parser ya regresa data_out final

        data_out, score_name = parser.merge_names(name_out, data_full)

        cnt = 0
        h = 20
        alpha_contrast = 1.8
        while score_name <= 1 and cnt <= 3:
            h -=4
            alpha_contrast-=0.3
            name_roi = processor.get_roi_name(crop, x1=30, y1=24, x2=42, y2=50, scale = 3.5, h_denoise= h,
                                          search_window_size = 21, alpha_contrast = alpha_contrast, beta_brightness = -10)
            texto_nombre = pytesseract.image_to_string(name_roi, lang="spa", config=config_name)
            name_out = parser.parse(texto_nombre)
            data_out, score_name = parser.merge_names(name_out, data_full)
            cnt+=1

        score = score_parse_result(data_out)
        # print(f"[CANDIDATO {i}] score={score}, data_out={data_out}")

        if score > best_score:
            best_score = score
            best_data = data_out

        # Si ya estamos bastante bien, podemos parar
        if score >= score_ok_threshold:
            best_data["attempt"] = f"yolo_candidate_{i}"
            best_data["score"] = score
            return best_data

    # Ninguno llegó al umbral, pero devolvemos el mejor que haya
    if best_data is not None:
        best_data["attempt"] = f"yolo_best_candidate"
        best_data["score"] = best_score
        return best_data

    # En teoría no deberías llegar aquí, pero por seguridad:
    return {
        "error": "No se pudo parsear ningún recorte de YOLO",
        "attempt": "yolo_candidates_failed",
    }

def close_on_space(event):
    if event.key == ' ':
        plt.close()

def main(ine_imagen):
    processor = IDImageProcessor(
        yolo_model_path=YOLO_PATH,
        conf_threshold=0.4,
        debug_dir="debug_dir",
        save_debug_images=False
    )

    # processed = processor.process_file(ine_imagen)
    # processed = processor.detect_and_mark_document(ine_imagen)
    processed = processor.get_document_crops(ine_imagen, page=0, max_candidates=2)
    # Extraer texto
    config = r"--psm 6 --oem 1 -c preserve_interword_spaces=1"
    # PSM 6 = texto linea por linea en horizontal, oem = interpretador, 0 es automatico
    # texto = pytesseract.image_to_string(processed, lang="spa", config=config)  # Cambia 'spa' según el idioma
    # print(texto)
    # # Visualizar rápido
    # test_roi = processor.get_roi_crop_percentage(processed, x1=0.10, x2=0.90, y1=0.1, y2=0.90)
    plt.figure(figsize=(8, 6))   # <-- tamaño en pulgadas
    plt.imshow(processed[0])
    plt.title("Imagen preprocesada para Tesseract")
    plt.axis("off")
    plt.gcf().canvas.mpl_connect('key_press_event', close_on_space)
    plt.show()

    test_roi = processor.get_roi_name(processed[0], x1=30, y1=24, x2=42, y2=50, scale = 3.5, h_denoise= 8,
                                          search_window_size = 21, alpha_contrast = 1.2, beta_brightness = -10)

    plt.figure(figsize=(8, 6))  # <-- tamaño en pulgadas
    plt.imshow(test_roi)
    plt.title("Imagen preprocesada para Tesseract")
    plt.gcf().canvas.mpl_connect('key_press_event', close_on_space)
    plt.axis("off")
    plt.show()
    # plt.imshow(processed, cmap="gray")
    # plt.imshow(test_roi)
    #
    # plt.title("Imagen recorrada para Tesseract")
    # plt.gcf().canvas.mpl_connect('key_press_event', close_on_space)
    # # plt.title("Imagen con debug")
    # plt.axis("off")
    # plt.show()

    # parser = INEParser()
    # data = parser.parse(texto)
    return 0
    # return json.dumps(data)

def main_v2(ine_imagen):
    processor = IDImageProcessor(
        yolo_model_path=YOLO_PATH,
        conf_threshold=0.4,
        debug_dir="debug_dir",
        save_debug_images=False
    )

    parser = INEParser()

    result  = process_with_yolo_candidates( processor, parser, ine_imagen)
    return json.dumps(result)


if __name__ == "__main__":
    # ine_imagen = "imagenes_prueba/INE_13.jpg"
    # ine_imagen = ("imagenes_prueba/INE_2.jpeg")
    ine_imagen = "imagenes_prueba/INEGloria.pdf"
    # ine_imagen = "imagenes_prueba/IneAdan.pdf"

    # print(main(ine_imagen))
    print(main_v2(ine_imagen))



