from image_processor import IDImageProcessor
import matplotlib.pyplot as plt
import pytesseract
from id_parser import INEParser
import json


def main():
    processor = IDImageProcessor(
        yolo_model_path="models/yolov8m.pt",
        conf_threshold=0.4,
        debug_dir="debug_dir",
        save_debug_images=False
    )

    # processed = processor.process_file("imagenes_prueba/INEGloria.pdf")
    # processed = processor.process_file("imagenes_prueba/INE_13.jpg")
    processed = processor.process_file("imagenes_prueba/IneAdan.pdf")

    # Extraer texto
    config = r"--psm 6 --oem 1 -c preserve_interword_spaces=1"
    # PSM 6 = texto linea por linea en horizontal
    texto = pytesseract.image_to_string(processed, lang="spa", config=config)  # Cambia 'spa' según el idioma
    # # Visualizar rápido
    plt.imshow(processed, cmap="gray")
    plt.title("Imagen preprocesada para Tesseract")
    plt.axis("off")
    plt.show()
    #
    parser = INEParser()
    data = parser.parse(texto)

    return json.dumps(data)


if __name__ == "__main__":
    print(main())





