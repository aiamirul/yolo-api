import os
import sys
import json
import glob
import base64
import mimetypes
import threading
import logging
from xml.etree import ElementTree as ET
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, render_template, send_file
from model_manager import ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("yolo-api")

app = Flask(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MAX_LOADED = int(os.environ.get("MAX_LOADED_MODELS", "5"))
manager = ModelManager(max_loaded=MAX_LOADED)

config = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        config = json.load(f)

DEFAULT_MODEL = config.get("default_model", config.get("model", "yolov8n"))

# --- Bookmarks ---
BOOKMARKS_FILE = os.path.join(os.path.dirname(__file__), "bookmarks.json")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample")
BOOKMARKS_LOCK = threading.Lock()


def load_bookmarks():
    if os.path.exists(BOOKMARKS_FILE):
        with open(BOOKMARKS_FILE) as f:
            return json.load(f)
    return {}


def save_bookmarks(bm):
    with open(BOOKMARKS_FILE, "w") as f:
        json.dump(bm, f, indent=2)


def get_sample_path(ext):
    ext = ext.lower()
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
        return os.path.join(SAMPLE_DIR, "sample.jpg")
    return os.path.join(SAMPLE_DIR, "sample.mp4")


def resolve_bookmark_path(bookmark):
    bm = load_bookmarks()
    if bookmark not in bm:
        return None, bm
    entry = bm[bookmark]
    base = os.path.join(os.path.dirname(__file__), entry["path"])
    pattern = entry.get("pattern", "**/*")
    full_pattern = os.path.join(base, pattern)
    return full_pattern, bm


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


def scan_models():
    built_in = [
        {"name": "yolov8n", "label": "yolov8n (nano)", "source": "built-in"},
        {"name": "yolov8s", "label": "yolov8s (small)", "source": "built-in"},
        {"name": "yolov8m", "label": "yolov8m (medium)", "source": "built-in"},
        {"name": "yolov8l", "label": "yolov8l (large)", "source": "built-in"},
        {"name": "yolov8x", "label": "yolov8x (xlarge)", "source": "built-in"},
    ]
    custom = []
    if os.path.isdir(MODELS_DIR):
        for pt in sorted(glob.glob(os.path.join(MODELS_DIR, "**", "*.pt"), recursive=True)):
            rel = os.path.relpath(pt, MODELS_DIR)
            parts = rel.split(os.sep)
            if len(parts) >= 2 and parts[-2] == "weights":
                name = parts[-3] if len(parts) >= 3 else parts[-4]
            else:
                name = os.path.splitext(parts[0])[0]
            custom.append({"name": name, "path": pt, "label": name + " (custom)", "source": "models/"})
    return built_in + custom


@app.route("/health", methods=["GET"])
def health():
    loaded = manager.list_loaded()
    return jsonify({
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "max_loaded_models": MAX_LOADED,
        "loaded_count": len(loaded),
        "loaded_models": [m["name"] for m in loaded],
    })


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": scan_models(), "default": DEFAULT_MODEL})


@app.route("/loaded", methods=["GET"])
def loaded_models():
    models = manager.list_loaded()
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = getattr(props, "total_mem", None) or getattr(props, "total_memory", 0)
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "total_vram": total,
                "allocated_vram": torch.cuda.memory_allocated(),
                "reserved_vram": torch.cuda.memory_reserved(),
            }
    except Exception:
        pass
    return jsonify({
        "models": models,
        "count": len(models),
        "max_loaded": MAX_LOADED,
        "gpu": gpu_info,
    })


@app.route("/load", methods=["POST"])
def load_model():
    data = request.get_json(force=True)
    name = data.get("model", "").strip()
    if not name:
        return jsonify({"error": "Missing 'model' field"}), 400

    if manager.is_loaded(name):
        return jsonify({"status": "ok", "model": name, "message": "Already loaded"})

    try:
        entry = manager.load(name)
        return jsonify({
            "status": "ok",
            "model": name,
            "path": entry["path"],
            "message": f"Model '{name}' loaded",
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500


@app.route("/unload", methods=["POST"])
def unload_model():
    data = request.get_json(force=True)
    name = data.get("model", "").strip()
    if not name:
        return jsonify({"error": "Missing 'model' field"}), 400

    if manager.unload(name):
        return jsonify({"status": "ok", "model": name, "message": f"Model '{name}' unloaded"})
    return jsonify({"error": f"Model '{name}' is not loaded"}), 404


@app.route("/gui", methods=["GET"])
def gui():
    return render_template("gui.html")


@app.route("/manage", methods=["GET"])
def manage_page():
    return render_template("manage.html")


@app.route("/settings", methods=["GET"])
def settings_page():
    return render_template("manage.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Missing 'image' field (base64 string)"}), 400

    model_name = data.get("model", "").strip() or None

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 data"}), 400

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        h, w = img.shape[:2]

        target = model_name or DEFAULT_MODEL
        yolo = manager.get(target)
        if yolo is None:
            entry = manager.load(target)
            yolo = entry["model"]

        results = yolo(img, verbose=False)

        objects = []
        for r in results:
            for box in r.boxes:
                objects.append({
                    "class": int(box.cls[0]),
                    "label": r.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": {
                        "x1": round(float(box.xyxy[0][0]), 2),
                        "y1": round(float(box.xyxy[0][1]), 2),
                        "x2": round(float(box.xyxy[0][2]), 2),
                        "y2": round(float(box.xyxy[0][3]), 2),
                    },
                })

        fmt = request.args.get("format", "json").lower()

        if fmt == "pascalvoc":
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = "image"
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(w)
            ET.SubElement(size, "height").text = str(h)
            ET.SubElement(size, "depth").text = str(img.shape[2] if len(img.shape) > 2 else 1)
            for obj in objects:
                o = ET.SubElement(root, "object")
                ET.SubElement(o, "name").text = obj["label"]
                ET.SubElement(o, "confidence").text = str(obj["confidence"])
                bndbox = ET.SubElement(o, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(obj["bbox"]["x1"]))
                ET.SubElement(bndbox, "ymin").text = str(int(obj["bbox"]["y1"]))
                ET.SubElement(bndbox, "xmax").text = str(int(obj["bbox"]["x2"]))
                ET.SubElement(bndbox, "ymax").text = str(int(obj["bbox"]["y2"]))
            xml_str = ET.tostring(root, encoding="unicode", xml_declaration=True)
            return Response(xml_str, mimetype="text/xml")

        return jsonify({
            "model": target,
            "count": len(objects),
            "objects": objects,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- File Manager Endpoints ---

@app.route("/files", methods=["GET"])
def files_page():
    return render_template("files.html")


@app.route("/getbookmarks", methods=["GET"])
def get_bookmarks():
    return jsonify({"bookmarks": load_bookmarks()})


@app.route("/bookmarks", methods=["POST"])
def add_bookmark():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    path = data.get("path", "").strip()
    pattern = data.get("pattern", "**/*").strip()
    label = data.get("label", "").strip() or name
    if not name or not path:
        return jsonify({"error": "Missing 'name' or 'path'"}), 400

    with BOOKMARKS_LOCK:
        bm = load_bookmarks()
        bm[name] = {"path": path, "pattern": pattern, "label": label}
        save_bookmarks(bm)
    log.info("Bookmark added: %s -> %s/%s", name, path, pattern)
    return jsonify({"status": "ok", "name": name})


@app.route("/bookmarks", methods=["DELETE"])
def delete_bookmark():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Missing 'name'"}), 400

    with BOOKMARKS_LOCK:
        bm = load_bookmarks()
        if name not in bm:
            return jsonify({"error": f"Bookmark '{name}' not found"}), 404
        del bm[name]
        save_bookmarks(bm)
    log.info("Bookmark deleted: %s", name)
    return jsonify({"status": "ok", "name": name})


@app.route("/getfile", methods=["GET"])
def get_file():
    bookmark = request.args.get("bookmark", "").strip()
    query = request.args.get("query", "").strip()
    limit = request.args.get("limit", 1, type=int)

    if not bookmark:
        return jsonify({"error": "Missing 'bookmark' parameter"}), 400

    full_pattern, bm = resolve_bookmark_path(bookmark)
    if full_pattern is None:
        ext = os.path.splitext(query)[1] if query else ".mp4"
        return jsonify({
            "status": "notfound",
            "bookmark": bookmark,
            "query": query,
            "sample": os.path.relpath(get_sample_path(ext), os.path.dirname(__file__)),
            "ext": ext,
        })

    try:
        matches = glob.glob(full_pattern, recursive=True)
    except Exception:
        matches = []

    if query:
        q = query.lower()
        matches = [m for m in matches if q in os.path.basename(m).lower()]

    matches.sort()

    if not matches:
        ext = os.path.splitext(query)[1] if query else ".mp4"
        return jsonify({
            "status": "notfound",
            "bookmark": bookmark,
            "query": query,
            "sample": os.path.relpath(get_sample_path(ext), os.path.dirname(__file__)),
            "ext": ext,
        })

    file_path = matches[0] if limit == 1 else matches[:limit]
    if limit == 1:
        mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        return send_file(file_path, mimetype=mime)

    results = []
    for fp in matches[:limit]:
        mime = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        results.append({
            "path": os.path.relpath(fp, os.path.dirname(__file__)),
            "name": os.path.basename(fp),
            "size": os.path.getsize(fp),
            "mime": mime,
        })
    return jsonify({"bookmark": bookmark, "query": query, "count": len(results), "files": results})


@app.route("/filecounts", methods=["GET"])
def file_counts():
    bookmark = request.args.get("bookmark", "").strip()
    query = request.args.get("query", "").strip()
    if not bookmark:
        return jsonify({"error": "Missing 'bookmark' parameter"}), 400

    full_pattern, bm = resolve_bookmark_path(bookmark)
    if full_pattern is None:
        return jsonify({"bookmark": bookmark, "total": 0, "filtered": 0, "query": query})

    try:
        all_files = glob.glob(full_pattern, recursive=True)
    except Exception:
        all_files = []

    total = len(all_files)
    filtered = total
    if query:
        q = query.lower()
        filtered = sum(1 for f in all_files if q in os.path.basename(f).lower())

    return jsonify({"bookmark": bookmark, "total": total, "filtered": filtered, "query": query})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", config.get("port", 5000)))
    try:
        manager.load(DEFAULT_MODEL)
    except Exception as e:
        log.error("Startup model load failed: %s. Continuing without model.", e)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "0") == "1")
