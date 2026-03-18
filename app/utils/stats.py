import json
import os
from threading import Lock


class StatsManager:
    def __init__(self, file_path="logs/usage_stats.json"):
        # Aseguramos ruta absoluta para evitar problemas con el directorio de trabajo de NSSM
        self.file_path = os.path.abspath(file_path)
        self.lock = Lock()
        self._ensure_file()

    def _ensure_file(self):
        # SI EL ARCHIVO YA EXISTE, NO HACE NADA. Si es un archivo nuevo, inicia en 0.
        if not os.path.exists(self.file_path):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            initial_stats = {
                "total_requests": 0,
                "total_processing_time": 0.0,  # Suma de todos los segundos
                "avg_response_time": 0.0,
                "engines": {"mistral": 0, "paddle": 0, "tesseract": 0},
                "outcomes": {"success": 0, "failure": 0}
            }
            with open(self.file_path, "w") as f:
                json.dump(initial_stats, f, indent=4)

    def log_usage(self, engine: str, success: bool, response_time: float):
        with self.lock:
            with open(self.file_path, "r") as f:
                stats = json.load(f)

            # Actualización de contadores
            stats["total_requests"] += 1
            stats["engines"][engine] = stats["engines"].get(engine, 0) + 1

            if success:
                stats["outcomes"]["success"] += 1
            else:
                stats["outcomes"]["failure"] += 1

            # Lógica de Latencia Promedio
            stats["total_processing_time"] += response_time
            stats["avg_response_time"] = stats["total_processing_time"] / stats["total_requests"]

            with open(self.file_path, "w") as f:
                json.dump(stats, f, indent=4)