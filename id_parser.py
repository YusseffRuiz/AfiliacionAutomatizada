import re
from typing import Dict, Optional, List


class INEParser:
    """
    Parser de texto de credenciales INE (formato actual).
    Toma el texto de Tesseract y devuelve un diccionario estructurado.
    """

    def __init__(self):
        # Regex precompilados
        self.re_curp = re.compile(r"\b[A-Z]{4}\d{6}[A-Z]{6}[A-Z0-9]{2}\b")
        self.re_clave_elector = re.compile(r"\b[A-Z0-9]{18}\b")
        self.re_fecha = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
        self.re_anio = re.compile(r"\b(19|20)\d{2}\b")
        self.re_vigencia = re.compile(
            r"\b((19|20)\d{2})\s*[-–]?\s*((19|20)\d{2})\b"
        )

    # --------- API pública ---------
    @staticmethod
    def merge_names(data_name, data_complete):
        def score_name(name_dict):
            score = 0
            if name_dict.get("apellido_paterno"):
                score += 1
            if name_dict.get("apellido_materno"):
                score += 1
            if name_dict.get("nombres"):
                score += 1
            return score

        score_data_name = score_name(data_name)
        # print(f"SCORE ROI: {score_data_name}")
        # print(data_name)
        score_data_complete = score_name(data_complete)-1 # Penalizacion, solo usar si hubo gran diferencia > 2
        # print(f"SCORE COMPLETE: {score_data_complete}")
        # print(data_complete)

        data_out = data_complete.copy()
        if score_data_complete>score_data_name:
            data_out["apellido_paterno"] = data_complete["apellido_paterno"]
            data_out["apellido_materno"] = data_complete["apellido_materno"]
            data_out["nombres"] = data_complete["nombres"]
            data_out["nombre_completo"] = data_complete["nombre_completo"]
            score = score_data_complete
        else:
            data_out["apellido_paterno"] = data_name["apellido_paterno"]
            data_out["apellido_materno"] = data_name["apellido_materno"]
            data_out["nombres"] = data_name["nombres"]
            data_out["nombre_completo"] = data_name["nombre_completo"]
            score = score_data_name
        return data_out, score

    def public_parse_name(self, raw_text: str) -> Dict:
        data = {
            "apellido_paterno": None,
            "apellido_materno": None,
            "nombres": None,
            "nombre_completo": None,
        }
        text = self._normalize_text(raw_text)
        lines = [l for l in text.split("\n") if l.strip()]
        self._parse_nombre(lines, data)
        return data

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

    @staticmethod
    def _letters_and_spaces(line: str) -> str:
        """
        Deja solo letras (incluyendo acentos y Ñ) y espacios.
        Colapsa espacios múltiples.
        """
        line = line.upper()
        line = re.sub(r"[^A-ZÁÉÍÓÚÑ ]+", " ", line)
        line = re.sub(r"\s+", " ", line)
        return line.strip()


    def _parse_nombre(self, lines: List[str], data: Dict):
        """
        Busca la sección NOMBRE y toma las 2–3 líneas siguientes
        como apellidos y nombres, pero limpiando ruido fuerte.
        texto en la MISMA línea tomando 1er token como apellido y sucesivamente
        """

        def _split_single_line_name(name_line: str):
            """
            Recibe algo como 'DE JESUS GARCIA GILBERTO' y devuelve
            (apellido_paterno, apellido_materno, nombres).

            Heurística:
            - Cada apellido puede tener hasta 2 tokens.
            - Si empieza con partícula (DE, DEL, LA, LOS, LAS, Y),
              se toma partícula + siguiente token como apellido.
            """
            tokens = name_line.split()
            if not tokens:
                return None, None, None

            PARTICULAS = {"DE", "DEL", "LA", "LAS", "LOS", "Y"}

            def build_surname(start_idx: int):
                """
                Construye un apellido a partir de tokens[start_idx],
                devolviendo (apellido_str, next_index).
                Máximo 2 tokens por apellido.
                """
                n = len(tokens)
                if start_idx >= n:
                    return None, start_idx

                t0 = tokens[start_idx]

                # Si empieza con partícula y hay al menos 2 tokens disponibles: tomar 2
                if t0 in PARTICULAS and start_idx + 1 < n:
                    apellido = f"{tokens[start_idx]} {tokens[start_idx + 1]}"
                    return apellido, start_idx + 2

                # Si no es partícula o no hay espacio para 2 tokens: tomar solo uno
                return t0, start_idx + 1

            # Apellido paterno
            ap_p, idx = build_surname(0)

            # Apellido materno
            ap_m, idx = build_surname(idx)

            # Nombres = lo que queda
            nombres = " ".join(tokens[idx:]) if idx < len(tokens) else None

            # Normalizar vacíos
            ap_p = ap_p if ap_p else None
            ap_m = ap_m if ap_m else None
            nombres = nombres if nombres else None

            return ap_p, ap_m, nombres


        idxs = self._find_line_indices(lines, "NOMBRE")
        if not idxs:
            idx = 0 # Suele ser la primera linea leida
        else:
            idx = idxs[0]
        raw_name_lines = []

        line_nombre = lines[idx].strip()
        m_inline = re.search(r"NOMBRE\s+(.+)$", line_nombre)
        if m_inline:
            inline_text = self._letters_and_spaces(m_inline.group(1).strip())
            if inline_text:
                raw_name_lines.append(inline_text)

        # Suele venir en las 3 líneas siguientes
        for i in range(idx + 1, min(idx + 7, len(lines))):
            line = lines[i].strip()
            if not line:
                continue
            line = line.upper()
            # Parar cuando aparezca otra etiqueta fuerte
            if any(tag in line for tag in ["DOMICILIO", "CLAVE DE ELECTOR", "CURP"]):
                break
            if line.startswith("SEX"):
                continue
            if "FECHA" in line or "NACIM" in line:
                continue

            raw_name_lines.append(line)

        if not raw_name_lines:
            return

        # Limpieza fuerte: solo letras y espacios
        clean_lines = []
        for ln in raw_name_lines:
            ln_clean = self._letters_and_spaces(ln)
            if ln_clean:
                clean_lines.append(ln_clean)

        if not clean_lines:
            return

        # Heurística: [0] paterno, [1] materno, [2+] nombres
        # Y nos quedamos con primeras palabras para quitar ruido residual

        # si solo hay UNA línea (caso Mistral) la dividimos en tokens
        if len(clean_lines) == 1:
            ap_p, ap_m, nombres = _split_single_line_name(clean_lines[0])
            if ap_p:
                data["apellido_paterno"] = ap_p
            if ap_m:
                data["apellido_materno"] = ap_m
            if nombres:
                data["nombres"] = nombres
            return

        if len(clean_lines) >= 1:
            tokens0 = clean_lines[0].split()
            tokens_validos = [t for t in tokens0 if (len(t) > 2 or t == 'DE' or t == "Y")]  # Limitamos ruido verificando si los apellidos tienen mas de 2 letras
            if tokens_validos:
                data["apellido_paterno"] = " ".join(tokens_validos[:3])

        if len(clean_lines) >= 2:
            tokens1 = clean_lines[1].split()
            tokens_validos = [t for t in tokens1 if ((len(t) > 2 or t == 'DE' or t == "Y") and not (
                        t == "SEXO" or t == "SEX"))]  # Limitamos ruido verificando si los nombres tienen mas de 2 letras
            if len(clean_lines) == 2:
                # Si solo hay 2 líneas, asumimos que la segunda son nombres
                if tokens_validos:
                    data["nombres"] = " ".join(tokens_validos[:3])
            else:
                if tokens_validos:
                    data["apellido_materno"] = " ".join(tokens_validos[:3])

        if len(clean_lines) >= 3:
            nombres_text = " ".join(clean_lines[2:])
            tokens_nombres = nombres_text.split()

            tokens_validos = [t for t in tokens_nombres if len(t) > 2] # Limitamos ruido verificando si los nombres tienen mas de 2 letras

            if tokens_validos:
                data["nombres"] = " ".join(tokens_validos[:3])

    @staticmethod
    def _parse_domicilio_pattern(lines: List[str]):
        dom_start = None
        max_lines = 3
        for i, line in enumerate(lines):
            l = line.strip()
            if re.search(r"^[\-\s]?\s*(CALLE|AV\.?|AVENIDA|BLVD\.?|BOULEVARD|PASAJE|ANDADOR|CDA|CERRADA|AV|LOC|C|COL|DEL\s)", l):
                dom_start = i
                break

        if dom_start is None:
            ##Failed
            return

        # 2) Acumular líneas hasta encontrar otra etiqueta fuerte
        dom_lines = []
        stop_tags = [
            "CLAVE DE ELECTOR",
            "ELECTOR",
            "CLAVE",
            "CURP",
            "FECHA DE NACIMIENTO",
            "SECCION",
            "SECCIÓN",
            "VIGENCIA",
            "AÑO DE REGISTRO",
        ]
        line_cnt = 0
        for j in range(dom_start, len(lines)):
            line_cnt += 1
            if line_cnt > max_lines:
                break
            line = lines[j].strip()
            if not line:
                continue

            # Si encontramos otra sección, dejamos de acumular
            if j != dom_start and any(tag in line for tag in stop_tags):
                break
            # Limpiar ruido: dejar solo letras, números, comas y espacios
            clean = line.upper()
            clean = re.sub(r"[^A-ZÁÉÍÓÚÑ0-9, ]+", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()

            # Eliminar tokens de 1 carácter (ruido tipo 'K')
            tokens = clean.split()
            tokens = [t for t in tokens if len(t) > 1 or "," in t]
            clean = " ".join(tokens)

            if clean:
                dom_lines.append(clean)

        return dom_lines

    def _parse_domicilio(self, lines: List[str], data: Dict):
        """
        Busca la sección DOMICILIO y toma varias líneas siguientes
        hasta encontrar otra etiqueta.
        """
        idxs = self._find_line_indices(lines, "DOMICILIO")
        noise_idx_flag = False
        stop_tags = [
            "CLAVE DE ELECTOR",
            "ELECTOR",
            "CLAVE",
            "CURP",
            "FECHA DE NACIMIENTO",
            # "SECCION",
            # "SECCIÓN",
            "VIGENCIA",
            "AÑO DE REGISTRO",
        ]
        if len(idxs) > 0 and idxs[0] >= (len(lines) - 3):
                noise_idx_flag = True
        if not idxs or noise_idx_flag:
            data["domicilio_lineas"] = self._parse_domicilio_pattern(lines)
        else:
            idx = idxs[0]
            dom_lines = []

            # 1) Contenido en la misma línea que 'DOMICILIO'
            line_dom = lines[idx].strip()
            m_inline = re.search(r"DOMICILIO\s+(.+)$", line_dom)

            if m_inline:
                first_dom = m_inline.group(1).strip()
                # limpiar basura visual
                first_dom = re.sub(r"\s{2,}.*$", "", first_dom).strip()
                first_dom = re.sub(r"^[\-\.\·\•\_\|\s]+", "", first_dom).strip()
                if first_dom:
                    dom_lines.append(first_dom)

            for i in range(idx + 1, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                # Parar cuando aparezcan otras etiquetas
                if any(tag in line for tag in stop_tags) or i>idx+3:
                    break
                    # Limpiar ruido: dejar solo letras, números, comas y espacios
                clean = line.upper()
                clean = re.sub(r"[^A-ZÁÉÍÓÚÑ0-9, ]+", " ", clean)
                clean = re.sub(r"\s+", " ", clean).strip()

                # Eliminar tokens de 1 carácter (ruido tipo 'K')
                tokens = clean.split()
                tokens = [t for t in tokens if len(t) > 1 or "," in t]
                clean = " ".join(tokens)

                if clean:
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