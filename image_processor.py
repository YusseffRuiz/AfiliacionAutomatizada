from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO
from PIL import Image
import uuid



class IDImageProcessor:
    """
    Clase interfaz de entrada:
    - Leer archivos de imagen (JPG/PNG/TIFF) o PDF. Pensado para soportar más tipos de archivos.
    - Detectar la credencial con YOLOv8 (estable y optimizado, propenso a pocos bugs).
    - Recortar y preprocesar la imagen para OCR (Salida de forma ideal para análisis).
    """

    SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tiff"}
    SUPPORTED_PDF_EXT = {".pdf"}

    def __init__(self, yolo_model_path: str, conf_threshold: float = 0.4, save_debug_images: bool = False,
        debug_dir: str = "debug_outputs",):
        """
        :param yolo_model_path: ruta al modelo YOLOv8 entrenado (ej. 'runs/detect/train/weights/best.pt')
        :param conf_threshold: umbral de confianza mínimo para detección
        :param save_debug_images: Salvar imagenes preprocesadas
        :param debug_dir: directorio para guardar debugs
        """
        self.model = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.save_debug_images = save_debug_images
        self.debug_dir = Path(debug_dir)

        if self.save_debug_images:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ---------- API pública ----------

    def process_file(self, path: str, page: int = 0) -> np.ndarray:
        """
        Lee el archivo, detecta la credencial y devuelve una imagen procesada
        (lista para Tesseract) como np.ndarray (grayscale).

        :param path: ruta al archivo (imagen o PDF)
        :param page: índice de página si es PDF (0 = primera)
        :return: imagen preprocesada en escala de grises
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()

        if ext in self.SUPPORTED_IMAGE_EXT:
            bgr = self._load_image(path_obj)
        elif ext in self.SUPPORTED_PDF_EXT:
            bgr = self._load_pdf_page_as_image(path_obj, page=page)
        else:
            supported = " ".join(self.SUPPORTED_IMAGE_EXT)
            raise ValueError(f"Formato no soportado: {ext}, favor de usar {supported} or pfdf")

        # 1) Detectar credencial con YOLO
        cropped = self._detect_and_crop_document(bgr)

        # 2) Preprocesar para Tesseract
        preprocessed = self._preprocess_for_ocr(cropped)

        # TODO Implementar sistema anti rotacion

        debug_id = None
        if self.save_debug_images:  ## Imagen pre recortada y en grises para debug
            # Reusar el mismo id si ya existe, para vincular original y recorte
            if debug_id is None:
                debug_id = self._generate_debug_id()
            self._save_debug_image(cropped, suffix="_cropped", debug_id=debug_id)
            self._save_debug_image(preprocessed, suffix="_gray", debug_id=debug_id)

        # Obtener ROI de la imagen, basado en las proporciones regulares de la INE.
        # Modificar de acuerdo a otros documentos.

        # h, w = preprocessed.shape  # processed = imagen gris grande
        #
        # x1 = int(0.30 * w) # Izquierda, entre mayor, mas se recorta.
        # y1 = int(0.22 * h) # Arriba, entre mayor, mas se recorta.
        # x2 = int(0.89 * w) # Derecha, entre menor, mas se recorta.
        # y2 = int(0.97 * h) # Abajo, entre menor, mas se recorta.
        #
        # roi = preprocessed[y1:y2, x1:x2]

        return cropped

    # ---------- Métodos internos ----------

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Carga JPG/PNG en BGR (OpenCV)."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        return img

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Ordena los 4 puntos como:
        [top-left, top-right, bottom-right, bottom-left]
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect


    def _warp_card_perspective(self, bgr: np.ndarray) -> np.ndarray:
        """
        Intenta encontrar el rectángulo de la credencial dentro del recorte
        y aplicar corrección de perspectiva. Si falla, regresa el recorte original.
        """
        # Trabajamos con una copia reducida para el análisis
        h, w = bgr.shape[:2]
        scale = 800.0 / max(h, w)  # escalar lado mayor a 800 px
        if scale < 1.0:
            small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = bgr.copy()
            scale = 1.0

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Bordes
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)

        # Contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Si no hay contornos, regresamos tal cual
            return bgr

        # Tomar el contorno más grande
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        card_cnt = None
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                card_cnt = approx
                break

        if card_cnt is None:
            # No encontramos cuadrilátero limpio, regresamos original
            return bgr

        # Volver a escalar puntos a coordenadas de la imagen original
        card_cnt = card_cnt.reshape(4, 2).astype("float32")
        card_cnt = card_cnt / scale

        rect = self._order_points(card_cnt)

        # Definir tamaño destino con aspecto de tarjeta (ancho > alto)
        dst_width = 2000
        aspect_ratio = 85.6 / 54.0  # aprox tarjeta tipo ID
        dst_height = int(dst_width / aspect_ratio)

        dst = np.array(
            [
                [0, 0],
                [dst_width - 1, 0],
                [dst_width - 1, dst_height - 1],
                [0, dst_height - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(bgr, M, (dst_width, dst_height))

        # Normalizar orientación: queremos más ancho que alto
        h_warp, w_warp = warped.shape[:2]
        if h_warp > w_warp:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    """
    def _warp_card_perspective(self, bgr: np.ndarray) -> np.ndarray:
        '''
        Intenta encontrar el rectángulo de la credencial dentro del recorte
        y aplicar corrección de perspectiva. Si falla, regresa el recorte original.
        Filtra contornos pequeños o con aspecto muy vertical (como la foto).
        '''
        h, w = bgr.shape[:2]
        roi_area = float(h * w)

        # Trabajar con versión reducida para análisis geométrico
        scale = 800.0 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = bgr.copy()
            scale = 1.0

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bgr  # fallback

        best_rect = None
        best_score = -1.0

        expected_ar = 85.6 / 54.0  # ~1.59 tarjeta tipo ID

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) != 4:
                continue

            # Área en coords reducidas -> escalar al tamaño original
            cnt_scaled = approx.reshape(4, 2).astype("float32") / scale
            area = cv2.contourArea(cnt_scaled)
            if area <= 0:
                continue

            area_ratio = area / roi_area
            # Descartar rectángulos demasiado pequeños (como la foto)
            if area_ratio < 0.35:
                continue

            # Aspect ratio del bounding box de ese cuadrilátero
            x, y, bw, bh = cv2.boundingRect(cnt_scaled.astype("int32"))
            if bh == 0:
                continue
            ar = bw / float(bh)

            # Queremos algo horizontal, parecido a 1.6
            if not (1.3 <= ar <= 2.2):
                continue

            # Score: combinación de área y proximidad al aspect ratio esperado
            score = area_ratio - abs(ar - expected_ar)

            if score > best_score:
                best_score = score
                best_rect = cnt_scaled

        if best_rect is None:
            # No encontramos buen candidato; usamos recorte original
            return bgr

        rect = self._order_points(best_rect)

        # Tamaño destino
        dst_width = 1000
        aspect_ratio = expected_ar
        dst_height = int(dst_width / aspect_ratio)

        dst = np.array(
            [
                [0, 0],
                [dst_width - 1, 0],
                [dst_width - 1, dst_height - 1],
                [0, dst_height - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(bgr, M, (dst_width, dst_height))

        # Asegurar orientación horizontal
        h_warp, w_warp = warped.shape[:2]
        if h_warp > w_warp:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped
        
    """

    @staticmethod
    def _load_pdf_page_as_image(path: Path, page: int = 0, dpi: int = 300) -> np.ndarray:
        """
        :param path: ruta al archivo (imagen o PDF)
        :param page: pagina a convertir
        :param dpi: pixeles a entregar
        Convierte una página de PDF en imagen (BGR).
        Por default, se usa la primera pagina, se puede indicar otra.
        """
        pages = convert_from_path(str(path), dpi=dpi)
        if not pages:
            raise ValueError(f"No se pudieron extraer páginas del PDF: {path}")
        if page >= len(pages):
            raise IndexError(f"El PDF tiene solo {len(pages)} páginas, pediste page={page}")

        pil_image = pages[page].convert("RGB")
        # PIL (RGB) -> OpenCV (BGR)
        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return bgr

    def _detect_and_crop_document(self, bgr: np.ndarray) -> np.ndarray:
        """
        Ejecuta YOLOv8 para detectar la credencial y devuelve el recorte.
        Usamos el bounding box con mayor confianza.
        """
        # YOLO espera imagen en formato estándar (BGR está bien, ultralytics lo maneja)
        results = self.model(bgr, conf=self.conf_threshold, verbose=False)

        if not results or len(results[0].boxes) == 0:
            raise RuntimeError("YOLOv8 no encontró ninguna credencial en la imagen.")

        boxes = results[0].boxes

        # Elegimos el box de mayor confianza
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[best_idx]

        # Asegurarnos de que los índices sean enteros y válidos
        h, w = bgr.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))


        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        cropped = bgr[y1:y2, x1:x2]

        if cropped.size == 0:
            raise RuntimeError("El recorte de la credencial está vacío, revisar detección.")

        warped = self._warp_card_perspective(cropped)
        return warped

    @staticmethod
    def _preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
        """
        Normaliza la imagen:
        - Convertir a gris
        - Redimensionar a un tamaño estándar
        - Binarización / mejora de contraste
        Devuelve imagen en escala de grises.
        """
        # A gris
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2) Quitar ruido sin destruir bordes
        #    (mejor que blur simple para texto),
        #    higher the H, higher the filter strenght. Cuidado que puede remover detalles finos.
        gray = cv2.fastNlMeansDenoising(gray, None, h=32, templateWindowSize=10, searchWindowSize=21)

        # 3) Aumentar contraste local (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 4) Reescalar agresivamente para que las letras sean grandes
        h, w = gray.shape
        scale = 3.0
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        return gray

    # ---------- Utilidades de debug ----------

    @staticmethod
    def _generate_debug_id() -> str:
        """Genera un ID único para vincular original y recorte."""
        return uuid.uuid4().hex[:4]

    def _save_debug_image(
            self,
            bgr: np.ndarray,
            suffix: str,
            debug_id: str | None = None,
    ) -> str:
        """
        Guarda imagen BGR como PNG en el directorio de debug.
        Devuelve el debug_id usado.
        """
        if debug_id is None:
            debug_id = self._generate_debug_id()

        filename = f"{debug_id}{suffix}.png"
        out_path = self.debug_dir / filename

        cv2.imwrite(str(out_path), bgr)
        return debug_id