import os
import time
import json
import threading
import logging
from collections import OrderedDict
from ultralytics import YOLO

log = logging.getLogger("yolo-api")

VRAM_FILE = os.path.join(os.path.dirname(__file__), "vram_usage.json")


def _get_vram_before():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except Exception:
        pass
    return None


def _get_vram_after():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except Exception:
        pass
    return None


def _load_vram_estimates():
    if os.path.exists(VRAM_FILE):
        try:
            with open(VRAM_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_vram_estimates(estimates):
    with open(VRAM_FILE, "w") as f:
        json.dump(estimates, f, indent=2)


class ModelManager:
    def __init__(self, max_loaded=5):
        self.max_loaded = max_loaded
        self._models = OrderedDict()  # name -> {"model": YOLO, "path": str, "loaded_at": float}
        self._lock = threading.Lock()
        self._vram_estimates = _load_vram_estimates()

    def _resolve_path(self, name):
        base = os.path.dirname(__file__)
        if os.path.isfile(name):
            return name
        if os.path.isfile(os.path.join(base, f"{name}.pt")):
            return os.path.join(base, f"{name}.pt")
        if os.path.isfile(f"{name}.pt"):
            return f"{name}.pt"
        models_dir = os.path.join(base, "models")
        if os.path.isdir(models_dir):
            for pt in _scan_models_dir(models_dir):
                if pt.endswith(f"{name}.pt") or os.path.basename(pt).replace(".pt", "") == name:
                    return pt
        return name

    def load(self, name):
        with self._lock:
            if name in self._models:
                self._models.move_to_end(name)
                log.info("Model '%s' already loaded, moving to front", name)
                return self._models[name]

            if len(self._models) >= self.max_loaded:
                evict_name, _ = self._models.popitem(last=False)
                log.info("Evicting model '%s' to make room", evict_name)

            path = self._resolve_path(name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            vram_before = _get_vram_before()
            log.info("Loading model '%s' from %s", name, path)
            yolo = YOLO(path)
            vram_after = _get_vram_after()

            if vram_before is not None and vram_after is not None:
                consumed = vram_after - vram_before
                self._vram_estimates[name] = consumed
                _save_vram_estimates(self._vram_estimates)
                log.info("Model '%s' loaded, VRAM used: %.1f MB", name, consumed / 1024 / 1024)
            else:
                log.info("Model '%s' loaded (VRAM tracking unavailable)", name)

            self._models[name] = {
                "model": yolo,
                "path": path,
                "loaded_at": time.time(),
            }
            return self._models[name]

    def unload(self, name):
        with self._lock:
            if name not in self._models:
                return False
            entry = self._models.pop(name)
            del entry["model"]
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            log.info("Model '%s' unloaded", name)
            return True

    def get(self, name=None):
        with self._lock:
            if name:
                if name in self._models:
                    self._models.move_to_end(name)
                    return self._models[name]["model"]
                return None
            if self._models:
                first_name = next(iter(self._models))
                return self._models[first_name]["model"]
            return None

    def is_loaded(self, name):
        with self._lock:
            return name in self._models

    def list_loaded(self):
        with self._lock:
            result = []
            for name, entry in self._models.items():
                result.append({
                    "name": name,
                    "path": entry["path"],
                    "loaded_at": entry["loaded_at"],
                    "vram_bytes": self._vram_estimates.get(name, 0),
                })
            return result

    def get_total_vram(self):
        with self._lock:
            return sum(self._vram_estimates.get(name, 0) for name in self._models)


def _scan_models_dir(models_dir):
    import glob
    results = []
    for pt in sorted(glob.glob(os.path.join(models_dir, "**", "*.pt"), recursive=True)):
        results.append(pt)
    return results
