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
    def get_document_crops(self, path: str, page: int = 0, max_candidates: int = 3) -> list[np.ndarray]:
        """
        Devuelve una lista de recortes (crops) de los bounding boxes
        detectados por YOLO, ordenados por confianza descendente.
        """
        bgr = self._load_bgr_from_path(path=path, page=page)

        results = self.model(bgr, conf=self.conf_threshold, verbose=False)

        if not results or len(results[0].boxes) == 0:
            raise RuntimeError("YOLOv8 no encontró ninguna credencial en la imagen.")

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # índices ordenados de mayor a menor confianza
        order = np.argsort(-confs)

        h, w = bgr.shape[:2]
        crops = []

        for idx in order[:max_candidates]:
            x1, y1, x2, y2 = xyxy[idx]

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            # margen pequeño alrededor
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            crop = bgr[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        if not crops:
            raise RuntimeError("No se pudieron obtener recortes válidos.")

        return crops

    def process_file(self, path: str, page: int = 0) -> np.ndarray:
        """
        Lee el archivo, detecta la credencial y devuelve una imagen procesada
        (lista para Tesseract) como np.ndarray (grayscale).

        :param path: ruta al archivo (imagen o PDF)
        :param page: índice de página si es PDF (0 = primera)
        :return: imagen preprocesada en escala de grises
        """
        bgr = self._load_bgr_from_path(path=path, page=page)

        # 1) Detectar credencial con YOLO
        cropped = self._detect_and_crop_document(bgr)

        # 2) Preprocesar para Tesseract
        preprocessed = self._preprocess_for_ocr(cropped, scale=3.0, h = 28, searchwindowssize=21, clahe_clip_limit=3.6,
                                                alpha_contrast=1.9, beta_brightness=-25)

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

        roi = self.get_roi_crop_percentage(preprocessed, x1=0.30, y1=0.22, x2=0.89, y2=0.97)

        return roi

    def pre_enhance_for_detection(self, path: str, page: int=0, scale: float=3.0) -> np.ndarray:
        """
        Mejora ligera de la imagen para ayudar a YOLO a detectar mejor la credencial.
        NO binariza; solo limpia ruido y aumenta contraste.
        """
        bgr = self._load_bgr_from_path(path=path, page=page)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # ruido ligero
        gray = cv2.fastNlMeansDenoising(gray, None, h=32, templateWindowSize=10, searchWindowSize=25)
        # contraste local
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # reescalar un poco si es muy chica
        h, w = gray.shape
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        # regresar en BGR para YOLO (duplicamos canal)
        enhanced_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 1) Detectar credencial con YOLO
        cropped = self._detect_and_crop_document(enhanced_bgr)

        # 2) Preprocesar para Tesseract
        preprocessed = self._preprocess_for_ocr(cropped, scale=1.0, h=15, searchwindowssize=7)

        return preprocessed

    def public_load_image(self, path: str, page: int=0):
        return self._load_bgr_from_path(path, page=page)

    def public_preprocess_for_ocr(self,bgr: np.ndarray=None, path: str=None, page: int=0, scale: float=3.0, h = 28,
                                  searchwindowssize=21, clahe_clip_limit=3.6, alpha_contrast=1.9,
                                  beta_brightness=-25) -> np.ndarray:
        if path is None and bgr is None:
            print("Error, you need either figure or path")
            return np.array([])
        if path is not None:
            bgr = self._load_bgr_from_path(path=path, page=page)
        preprocessed = self._preprocess_for_ocr(bgr, scale=scale, h = h, searchwindowssize=searchwindowssize,
                                                clahe_clip_limit=clahe_clip_limit,
                                                alpha_contrast=alpha_contrast, beta_brightness=beta_brightness)

        h, w = preprocessed.shape  # processed = imagen gris grande

        x1 = int(0.30 * w)  # Izquierda, entre mayor, mas se recorta.
        y1 = int(0.22 * h)  # Arriba, entre mayor, mas se recorta.
        x2 = int(0.89 * w)  # Derecha, entre menor, mas se recorta.
        y2 = int(0.97 * h)  # Abajo, entre menor, mas se recorta.

        roi = preprocessed[y1:y2, x1:x2]

        return roi

    def detect_and_mark_document(self, path: str, page: int=0) -> np.ndarray:
        """
        Ejecuta YOLOv8 para detectar la credencial dibujar un recuadro alrededor.
        Usamos el bounding box con mayor confianza.
        """
        bgr = self._load_bgr_from_path(path=path, page=page)
        # YOLO espera imagen en formato estándar (BGR está bien, ultralytics lo maneja)
        results = self.model(bgr, conf=self.conf_threshold, verbose=False)

        if not results or len(results[0].boxes) == 0:
            raise RuntimeError("YOLOv8 no encontró ninguna credencial en la imagen.")

        boxes = results[0].boxes

        # Elegimos el box de mayor confianza
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        classes = boxes.cls.cpu().numpy().astype(int)

        names = self.model.names # Dict: {0: 'birthday', 1: 'id', 2: 'name'}

        # Filtrar solo detecciones cuya clase sea 'idcard' (ajusta al nombre real)
        target_class_ids = [
            cid for cid, name in names.items()
            if name.lower() in "id"
        ]

        if not target_class_ids:
            # Si el modelo solo tiene una clase o no sabemos el nombre, usamos lo que hay
            target_class_ids = list(set(classes))

        candidate_indices = [
            i for i, cls_id in enumerate(classes)
            if cls_id in target_class_ids
        ]

        if not candidate_indices:
            raise RuntimeError("El modelo no detectó ninguna ID card, solo otras clases.")

        # Elegimos el candidato con mayor confianza entre los de clase válida
        best_idx = max(candidate_indices, key=lambda i: confs[i])

        # best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = xyxy[best_idx]

        # Limpiar indices
        h, w = bgr.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        # Dibujar rectángulo
        boxed = bgr.copy()
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Rojo, grosor 4 px

        # Opcional: dibujar el confidence score
        label = f"{names[classes[best_idx]]}: {confs[best_idx]:.2f}"
        cv2.putText(boxed, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        return boxed

    @staticmethod
    def get_roi_crop_percentage(bgr: np.ndarray, x1: float, x2:float, y1:float, y2:float) -> np.ndarray:
        """
        :param bgr: input image
        :param x1: Izquierda, entre mayor, mas se recorta.
        :param x2: Derecha, entre menor, mas se recorta.
        :param y1: Arriba, entre mayor, mas se recorta.
        :param y2: Abajo, entre menor, mas se recorta.
        :return: imagen recortada
        """
        h, w = bgr.shape  # processed = imagen gris grande

        x1 = int(x1 * w)  #
        y1 = int(y1 * h)  #
        x2 = int(x2 * w)  #
        y2 = int(y2 * h)  #

        roi = bgr[y1:y2, x1:x2]
        return roi

    def get_roi_name(self, bgr, x1=2, y1=3, x2=55, y2=63, scale: float = 3.5,
                    h_denoise: int = 18,
                    search_window_size: int = 21,
                    alpha_contrast: float = 1.8,
                    beta_brightness: int = -20,):
        """
                Preprocesado específico para el bloque de NOMBRE:
                - Gris
                - Denoising suave
                - CLAHE
                - Normalización + aumento de contraste
                - Upscale fuerte
                Devuelve imagen en escala de grises.
        """
        roi = self.get_roi_crop_margin(bgr, x1=x1, y1=y1, x2=x2, y2=y2)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=h_denoise,
            templateWindowSize=12,
            searchWindowSize=search_window_size,
        )
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Normalizar y subir contraste para separar letras del fondo
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.convertScaleAbs(gray, alpha=alpha_contrast, beta=beta_brightness)

        # Reescalar para que las letras sean grandes
        h_img, w_img = gray.shape
        gray = cv2.resize(
            gray,
            (int(w_img * scale), int(h_img * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

        # cv2.imshow('Image Title', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return gray



    @staticmethod
    def get_roi_crop_margin(bgr: np.ndarray, margin = 0, x1=0, x2=0, y1=0, y2=0) -> np.ndarray:
        """
        La entreda de los datos, se convierte a porcentaje para recortar margenes escalado a los pixeles de la imagen.
        Acepta de 0-100
        :param bgr: input image
        :param margin: recorte parejo
        :param x1: Recorte desde izquierda
        :param x2: Recorte desde Derecha
        :param y1: Recorde desde Arriba
        :param y2: Recorte desde abajo
        :return: imagen recortada
        """
        def margin_to_pixels(value, max_value):
            return int(value*max_value/100)
        h, w = bgr.shape[:2]

        if margin>0:
            margin_h = margin_to_pixels(margin, h)
            margin_w = margin_to_pixels(margin, w)
            x1 = max(0, margin_h)
            y1 = max(0, margin_w)
            x2 = min(w, w - margin_h)
            y2 = min(h, h - margin_w)
            return bgr[y1:y2, x1:x2]
        else:
            x1 = margin_to_pixels(x1, w)
            y1 = margin_to_pixels(y1, h)
            x2 = margin_to_pixels(x2, w)
            y2 = margin_to_pixels(y2, h)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, w - x2)
            y2 = min(h, h - y2)
            return bgr[y1:y2, x1:x2]

    # ---------- Métodos internos ----------
    def _load_bgr_from_path(self, path: str, page: int = 0) -> np.ndarray:
        """
        Carga cualquier archivo soportado (imagen o PDF) y regresa BGR.
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()

        if ext in self.SUPPORTED_IMAGE_EXT:
            bgr = self._load_image(path_obj)
        elif ext in self.SUPPORTED_PDF_EXT:
            bgr = self._load_pdf_page_as_image(path_obj, page=page)
        else:
            supported = " ".join(self.SUPPORTED_IMAGE_EXT)
            raise ValueError(f"Formato no soportado: {ext}, favor de usar {supported} o pdf")

        return bgr

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Carga JPG/PNG en BGR (OpenCV)."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        return img

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


        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        cropped = bgr[y1:y2, x1:x2]

        if cropped.size == 0:
            raise RuntimeError("El recorte de la credencial está vacío, revisar detección.")

        return cropped

    @staticmethod
    def _preprocess_for_ocr(bgr: np.ndarray, scale=3.0, h = 32, searchwindowssize=21, clahe_clip_limit=2.0,
                            alpha_contrast: float = 1.7, beta_brightness: int = -20,
                            ) -> np.ndarray:
        """
        Normaliza la imagen para Tesseract:
        - Convertir a gris
        - Denoising suave
        - Aumentar contraste local (CLAHE)
        - Aumentar contraste global de letras
        - Reescalar
        Devuelve imagen en escala de grises.
        """
        # 1) A gris
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2) Quitar ruido sin destruir bordes
        #    h: fuerza del filtro; no lo pongas demasiado alto para no borrar letras.
        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=h,
            templateWindowSize=12,
            searchWindowSize=searchwindowssize,
        )

        # 3) Aumentar contraste local (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 4) Aumentar contraste de letras vs fondo
        # 4a) Normalización de histograma completo
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # 4b) Contraste lineal: I' = alpha * I + beta
        #    alpha > 1 aumenta contraste, beta negativo oscurece un poco el fondo.
        gray = cv2.convertScaleAbs(gray, alpha=alpha_contrast, beta=beta_brightness)

        # 5) Reescalar agresivamente para que las letras sean grandes
        h_img, w_img = gray.shape
        gray = cv2.resize(
            gray,
            (int(w_img * scale), int(h_img * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

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