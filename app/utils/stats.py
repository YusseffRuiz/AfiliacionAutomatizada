import datetime
import json
import os
from threading import Lock


class StatsManager:
    def __init__(self, file_path="logs/stats/"):
        # Aseguramos ruta absoluta para evitar problemas con el directorio de trabajo de NSSM
        self.file_path = os.path.abspath(file_path)
        self.current_file = os.path.join(self.file_path, "usage_stats_global.json")
        self.lock = Lock()
        os.makedirs(self.file_path, exist_ok=True)
        self._ensure_file(self.current_file)

    def _get_monthly_path(self):
        now = datetime.datetime.now()
        return os.path.join(self.file_path, f"stats_{now.year}_{now.month:02d}.json")

    @staticmethod
    def _ensure_file(path: str):
        # SI EL ARCHIVO YA EXISTE, NO HACE NADA. Si es un archivo nuevo, inicia en 0.
        if not os.path.exists(path):
            initial_stats = {
                "total_requests": 0,
                "total_processing_time": 0.0,  # Suma de todos los segundos
                "avg_response_time": 0.0,
                "engines": {"mistral": 0, "paddle": 0, "tesseract": 0},
                "outcomes": {"success": 0, "failure": 0},
                "last_update": str(datetime.datetime.now())
            }
            with open(path, "w") as f:
                json.dump(initial_stats, f, indent=4)

    def log_usage(self, engine: str, success: bool, response_time: float):
        with self.lock:
            monthly_file = self._get_monthly_path()
            self._ensure_file(monthly_file)

            # Actualizamos AMBOS archivos: el Global y el Mensual
            for path in [self.current_file, monthly_file]:
                with open(path, "r") as f:
                    stats = json.load(f)

                stats["total_requests"] += 1
                stats["engines"][engine] = stats["engines"].get(engine, 0) + 1

                if success:
                    stats["outcomes"]["success"] += 1
                else:
                    stats["outcomes"]["failure"] += 1

                stats["total_processing_time"] += response_time
                stats["avg_response_time"] = stats["total_processing_time"] / stats["total_requests"]
                stats["last_update"] = str(datetime.datetime.now())

                with open(path, "w") as f:
                    json.dump(stats, f, indent=4)