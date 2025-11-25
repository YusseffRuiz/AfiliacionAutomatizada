import re
from typing import Dict, Optional, List


class INEParser:
    """
    Parser de texto de credenciales INE (formato actual).
    Toma el texto de Tesseract y devuelve un diccionario estructurado.
    """

    def __init__(self):
        # Regex precompilados
        self.re_curp = re.compile(r"\b[A-Z]{4}\d{6}[A-Z]{6}\d{2}\b")
        self.re_clave_elector = re.compile(r"\b[A-Z0-9]{18}\b")
        self.re_fecha = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
        self.re_anio = re.compile(r"\b(19|20)\d{2}\b")
        self.re_vigencia = re.compile(
            r"\b((19|20)\d{2})\s*[-–]?\s*((19|20)\d{2})\b"
        )

    # --------- API pública ---------

    def parse(self, raw_text: str) -> Dict:
        """
        Punto de entrada: recibe el texto crudo de Tesseract y devuelve un dict.
        """
        text = self._normalize_text(raw_text)
        lines = [l for l in text.split("\n") if l.strip()]

        data = {
            "apellido_paterno": None,
            "apellido_materno": None,
            "nombres": None,
            "nombre_completo": None,
            "sexo": None,
            "domicilio_lineas": [],
            "domicilio": None,
            "clave_elector": None,
            "curp": None,
            "fecha_nacimiento": None,
            # "seccion": None,
            "vigencia": None,
            # "anio_registro": None,
            # "raw_text": raw_text,
        }

        # Secciones basadas en etiquetas
        self._parse_nombre(lines, data)
        self._parse_domicilio(lines, data)
        self._parse_campos_por_regex(lines, data)
        self._parse_sexo(lines, data)

        # Post-procesar domicilio
        if data["domicilio_lineas"]:
            data["domicilio"] = ", ".join(data["domicilio_lineas"])

        # Construir nombre completo si se pudo extraer por partes
        parts = [
            data["apellido_paterno"],
            data["apellido_materno"],
            data["nombres"],
        ]
        if any(parts):
            data["nombre_completo"] = " ".join(p for p in parts if p)

        if not data["fecha_nacimiento"] and data["curp"]:
            inferred = self._infer_fecha_nacimiento_from_curp(data["curp"])
            if inferred:
                data["fecha_nacimiento"] = inferred

        if not data["sexo"] and data["curp"]:
            sex = self._infer_sexo_from_curp(data["curp"])
            if sex:
                data["sexo"] = sex

        data_out = { # Agregar o quitar campos segun sea la necesidad
            "apellido_paterno": data["apellido_paterno"],
            "apellido_materno": data["apellido_materno"],
            "nombres": data["nombres"],
            "nombre_completo": data["nombre_completo"],
            "sexo": ("HOMBRE" if data["sexo"] == "H" else "MUJER") if data.get("sexo") in ("H", "M") else None,
            "domicilio": data["domicilio"],
            "clave_elector": data["clave_elector"],
            "curp": data["curp"],
            "fecha_nacimiento": data["fecha_nacimiento"],
            "vigencia": data["vigencia"],
        }
        return data_out

    # --------- Normalización ---------

    @staticmethod
    def _normalize_text(text: str) -> str:
        # Uppercase, quitar caracteres raros, normalizar espacios
        text = text.replace("\r", "\n")
        text = text.upper()

        # Reemplazar caracteres no imprimibles por espacio
        text = re.sub(r"[^\x20-\x7E\n]", " ", text)

        # Colapsar espacios múltiples
        # text = re.sub(r"[ \t]+", " ", text)

        # Quitar espacios al final de línea
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text

    # --------- Parsing por secciones ---------

    @staticmethod
    def _find_line_indices(lines: List[str], keyword: str) -> List[int]:
        """Devuelve índices de líneas que contienen la palabra clave."""
        out = []
        for i, line in enumerate(lines):
            if keyword in line:
                out.append(i)
        return out

    def _parse_nombre(self, lines: List[str], data: Dict):
        """
        Busca la sección NOMBRE y toma las 2–3 líneas siguientes
        como apellidos y nombres.
        """

        def _clean_name_line(line: str) -> str:
            """
            Limpia una línea de nombre/apellido:
            - Elimina basura inicial
            - Elimina basura final
            """

            line = line.strip()

            # Caso 1: basura al inicio (token corto + muchos espacios + texto bueno)
            m = re.match(r"^([0-9A-ZÁÉÍÓÚÑ]{1,2})\s{2,}([A-ZÁÉÍÓÚÑ ].+)$", line)
            if m:
                line = m.group(2).strip()

            # Caso 2: basura al final (texto bueno + muchos espacios + token corto)
            m2 = re.match(r"^(.+?[A-ZÁÉÍÓÚÑ])\s{2,}[0-9A-ZÁÉÍÓÚÑ|]{1,3}$", line)
            if m2:
                line = m2.group(1).strip()

            return line
        idxs = self._find_line_indices(lines, "NOMBRE")
        if not idxs:
            return

        idx = idxs[0]
        name_lines = []

        # Suele venir en las 3 líneas siguientes
        for i in range(idx + 1, min(idx + 4, len(lines))):
            line = lines[i].strip()
            # Parar cuando aparezca otra etiqueta fuerte
            if any(tag in line for tag in ["DOMICILIO", "CLAVE DE ELECTOR", "CURP"]):
                break
            if line:
                line = _clean_name_line(line)
                if not line:
                    continue
                name_lines.append(line)

        if not name_lines:
            return

        # Heurística típica INE:
        # [0] apellido paterno
        # [1] apellido materno
        # [2:] nombres
        if len(name_lines) >= 1:
            data["apellido_paterno"] = name_lines[0]
        if len(name_lines) >= 2:
            data["apellido_materno"] = name_lines[1]
        if len(name_lines) >= 3:
            data["nombres"] = " ".join(name_lines[2:])
        elif len(name_lines) == 2:
            # Si solo hay 2 líneas, asumir que la segunda son nombres
            data["nombres"] = name_lines[1]

    @staticmethod
    def _parse_domicilio_pattern(lines: List[str], data: Dict):
        dom_start = None
        for i, line in enumerate(lines):
            l = line.strip()
            if re.match(r"^[\- ]?\s*(CALLE|AV\.?|AVENIDA|BLVD\.?|BOULEVARD|PASAJE|ANDADOR)\b", l):
                dom_start = i
                break

        if dom_start is None:
            ##Failed
            return

        # 2) Acumular líneas hasta encontrar otra etiqueta fuerte
        dom_lines = []
        stop_tags = [
            "CLAVE DE ELECTOR",
            "CURP",
            "FECHA DE NACIMIENTO",
            "SECCION",
            "SECCIÓN",
            "VIGENCIA",
            "AÑO DE REGISTRO",
        ]

        for j in range(dom_start, len(lines)):
            line = lines[j].strip()
            if not line:
                continue

            # Si encontramos otra sección, dejamos de acumular
            if j != dom_start and any(tag in line for tag in stop_tags):
                break
            clean = re.sub(r"\s{2,}.*$", "", line).strip()
            clean = re.sub(r"^[\-\.\·\•\_\|\s]+", "", clean).strip()
            dom_lines.append(clean)

        return dom_lines

    def _parse_domicilio(self, lines: List[str], data: Dict):
        """
        Busca la sección DOMICILIO y toma varias líneas siguientes
        hasta encontrar otra etiqueta.
        """
        idxs = self._find_line_indices(lines, "DOMICILIO")
        if not idxs:
            data["domicilio_lineas"] = self._parse_domicilio_pattern(lines, data)
        else:
            idx = idxs[0]
            dom_lines = []
            for i in range(idx + 1, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                # Parar cuando aparezcan otras etiquetas
                if any(tag in line for tag in ["CLAVE DE ELECTOR", "CURP", "FECHA DE NACIMIENTO"]):
                    break
                clean = re.sub(r"\s{2,}.*$", "", line).strip()
                clean = re.sub(r"^[\-\.\·\•\_\|\s]+", "", clean).strip()
                dom_lines.append(clean)

            data["domicilio_lineas"] = dom_lines

    @staticmethod
    def _parse_sexo(lines: List[str], data: Dict):
        """
        Intenta extraer el sexo desde el texto OCR.
        Busca líneas que contengan 'SEXO' y captura 'H' o 'M' en esa línea
        o en la siguiente.
        """
        if data.get("sexo"):
            return  # ya lo tenemos

        for i, line in enumerate(lines):
            if "SEXO" not in line:
                continue

            # 1) Intentar en la misma línea
            m = re.search(r"SEXO\s*([HM])\b", line)
            if m:
                data["sexo"] = m.group(1)
                return

            # 2) Si no está explícito, mirar la siguiente línea por un H/M aislado
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                m2 = re.match(r"^([HM])\b", next_line)
                if m2:
                    data["sexo"] = m2.group(1)
                    return

    # --------- Parsing por regex ---------

    def _parse_campos_por_regex(self, lines: List[str], data: Dict):
        joined = "\n".join(lines)

        # CURP
        m = self.re_curp.search(joined)
        if m:
            data["curp"] = m.group(0)

        # Clave de elector
        m = self.re_clave_elector.search(joined)
        if m:
            data["clave_elector"] = m.group(0)

        # Fecha de nacimiento: tomar la primera fecha que aparezca
        m = self.re_fecha.search(joined)
        if m:
            data["fecha_nacimiento"] = m.group(1)

        # Vigencia: buscar patrón "2021 - 2031" o similar
        m = self.re_vigencia.search(joined)
        if m:
            data["vigencia"] = f"{m.group(1)}-{m.group(3)}"
        else:
            # fallback: encontrar dos años seguidos
            years = self.re_anio.findall(joined)
            if len(years) >= 2:
                data["vigencia"] = f"{years[-2]}-{years[-1]}"

        # # Sección: línea que contenga la palabra SECCION o SECCIÓN + número
        # for line in lines:
        #     if "SECCION" in line or "SECCIÓN" in line:
        #         nums = re.findall(r"\b\d{3,4}\b", line)
        #         if nums:
        #             data["seccion"] = nums[-1]
        #             break
        #
        # # Año de registro: linea con "AÑO DE REGISTRO"
        # for line in lines:
        #     if "AÑO DE REGISTRO" in line:
        #         years = self.re_anio.findall(line)
        #         if years:
        #             # tomar el primer año válido
        #             data["anio_registro"] = years[0]
        #         else:
        #             # fallback: agarrar pares tipo "10 01" -> 2010
        #             nums = re.findall(r"\b\d{2}\b", line)
        #             if len(nums) >= 2:
        #                 data["anio_registro"] = f"20{nums[0]}"
        #         break


    # ------- Parsing infiriendo por curp --------

    @staticmethod
    def _infer_fecha_nacimiento_from_curp(curp: str):
        """
        Extrae la fecha de nacimiento del CURP (YYMMDD) y la vuelve DD/MM/YYYY.
        CURP: LLLL YY MM DD H ...
        """
        if not curp or len(curp) < 10:
            return None

        try:
            yy = int(curp[4:6])
            mm = int(curp[6:8])
            dd = int(curp[8:10])

            # Determinar siglo:
            # CURP usa:
            #  00-21 = 2000-2021 (o más, según año actual)
            #  22-99 = 1922-1999
            if yy <= 21:  # puedes aumentar este límite según necesidad
                yyyy = 2000 + yy
            else:
                yyyy = 1900 + yy

            return f"{dd:02d}/{mm:02d}/{yyyy}"
        except:
            return None

    @staticmethod
    def _infer_sexo_from_curp(curp: str) -> Optional[str]:
        """
        Extrae sexo de la CURP.
        Posición 11 (índice 10): H = hombre, M = mujer.
        Devuelve 'H' o 'M'.
        """
        if not curp or len(curp) < 11:
            return None
        c = curp[10].upper()
        if c in ("H", "M"):
            return c
        return None