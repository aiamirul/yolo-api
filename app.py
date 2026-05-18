import os
import sys
import json
import glob
import base64
import uuid
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
ALDATA_DIR = os.path.join(os.path.dirname(__file__), "alldata")
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
    bm = load_bookmarks()
    result = {}
    for name, entry in bm.items():
        base = os.path.join(os.path.dirname(__file__), entry["path"])
        pattern = entry.get("pattern", "**/*")
        full_pattern = os.path.join(base, pattern)
        try:
            matches = glob.glob(full_pattern, recursive=True)
            matches = [f for f in matches if os.path.isfile(f)]
        except Exception:
            matches = []
        total_size = sum(os.path.getsize(f) for f in matches)
        result[name] = {**entry, "count": len(matches), "total_size": total_size}
    return jsonify({"bookmarks": result})


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


@app.route("/upload", methods=["POST"])
def upload_file():
    bookmark = request.form.get("bookmark", "").strip()
    if not bookmark:
        return jsonify({"error": "Missing 'bookmark' field"}), 400

    bm = load_bookmarks()
    if bookmark not in bm:
        return jsonify({"error": f"Bookmark '{bookmark}' not found"}), 404

    base = os.path.join(os.path.dirname(__file__), bm[bookmark]["path"])
    os.makedirs(base, exist_ok=True)

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files provided"}), 400

    saved = []
    for f in files:
        if f.filename == "":
            continue
        safe_name = os.path.basename(f.filename)
        dest = os.path.join(base, safe_name)
        f.save(dest)
        saved.append({"name": safe_name, "size": os.path.getsize(dest)})
        log.info("Uploaded %s -> %s", safe_name, dest)

    return jsonify({"status": "ok", "bookmark": bookmark, "saved": len(saved), "files": saved})


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
        return jsonify({"bookmark": bookmark, "total": 0, "filtered": 0, "total_size": 0, "query": query})

    try:
        all_files = glob.glob(full_pattern, recursive=True)
        all_files = [f for f in all_files if os.path.isfile(f)]
    except Exception:
        all_files = []

    total = len(all_files)
    total_size = sum(os.path.getsize(f) for f in all_files)
    filtered = total
    filtered_size = total_size
    if query:
        q = query.lower()
        matched = [f for f in all_files if q in os.path.basename(f).lower()]
        filtered = len(matched)
        filtered_size = sum(os.path.getsize(f) for f in matched)

    return jsonify({
        "bookmark": bookmark,
        "total": total,
        "filtered": filtered,
        "total_size": total_size,
        "filtered_size": filtered_size,
        "query": query,
    })


# --- Data Save / Retrieve ---

@app.route("/alldata_ui", methods=["GET"])
def alldata_ui():
    return render_template("alldata.html")


@app.route("/alldata_folders", methods=["GET"])
def alldata_folders():
    os.makedirs(ALDATA_DIR, exist_ok=True)
    entries = []
    # Root-level files
    root_files = [f for f in glob.glob(os.path.join(ALDATA_DIR, "*.*"))
                  if os.path.isfile(f) and (f.endswith(".json") or f.endswith(".xml"))]
    if root_files:
        total_size = sum(os.path.getsize(f) for f in root_files)
        entries.append({"name": "(root)", "count": len(root_files), "total_size": total_size})
    # Subfolders
    for d in sorted(os.listdir(ALDATA_DIR)):
        full = os.path.join(ALDATA_DIR, d)
        if os.path.isdir(full):
            files = [f for f in glob.glob(os.path.join(full, "**", "*.*"), recursive=True)
                     if os.path.isfile(f) and (f.endswith(".json") or f.endswith(".xml"))]
            total_size = sum(os.path.getsize(f) for f in files)
            entries.append({"name": d, "count": len(files), "total_size": total_size})
    return jsonify({"folders": entries, "total_files": sum(e["count"] for e in entries)})

def _to_pascalvoc(data):
    root = ET.Element("annotation")
    if "filename" in data:
        ET.SubElement(root, "filename").text = str(data["filename"])
    if "folder" in data:
        ET.SubElement(root, "folder").text = str(data["folder"])
    if "size" in data and isinstance(data["size"], dict):
        size_el = ET.SubElement(root, "size")
        for k in ("width", "height", "depth"):
            if k in data["size"]:
                ET.SubElement(size_el, k).text = str(data["size"][k])
    if "source" in data and isinstance(data["source"], dict):
        src = ET.SubElement(root, "source")
        for k, v in data["source"].items():
            ET.SubElement(src, k).text = str(v)
    objects = data.get("objects", data.get("object", []))
    if isinstance(objects, dict):
        objects = [objects]
    for obj in objects:
        o = ET.SubElement(root, "object")
        if "name" in obj:
            ET.SubElement(o, "name").text = str(obj["name"])
        if "confidence" in obj:
            ET.SubElement(o, "confidence").text = str(obj["confidence"])
        if "pose" in obj:
            ET.SubElement(o, "pose").text = str(obj["pose"])
        if "truncated" in obj:
            ET.SubElement(o, "truncated").text = str(obj["truncated"])
        if "difficult" in obj:
            ET.SubElement(o, "difficult").text = str(obj["difficult"])
        if "bndbox" in obj and isinstance(obj["bndbox"], dict):
            bb = obj["bndbox"]
            bndbox = ET.SubElement(o, "bndbox")
            for k in ("xmin", "ymin", "xmax", "ymax"):
                if k in bb:
                    ET.SubElement(bndbox, k).text = str(int(bb[k]) if isinstance(bb[k], float) else bb[k])
        elif "bbox" in obj and isinstance(obj["bbox"], dict):
            bb = obj["bbox"]
            bndbox = ET.SubElement(o, "bndbox")
            keys_map = {"x1": "xmin", "y1": "ymin", "x2": "xmax", "y2": "ymax"}
            for sk, ek in keys_map.items():
                if sk in bb:
                    ET.SubElement(bndbox, ek).text = str(int(bb[sk]) if isinstance(bb[sk], float) else bb[sk])
    for k, v in data.items():
        if k not in ("filename", "folder", "size", "source", "objects", "object"):
            if not isinstance(v, (dict, list)):
                ET.SubElement(root, k).text = str(v)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


@app.route("/alldata", methods=["POST"])
def save_data():
    tag = request.args.get("TAG", "").strip()
    folder = request.args.get("FOLDER", "").strip()
    filetype = request.args.get("filetype", "JSON").strip().upper()

    if not tag:
        return jsonify({"error": "Missing ?TAG= parameter"}), 400

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    uid = uuid.uuid4().hex[:12]
    filename = f"{uid}_{tag}"

    save_dir = ALDATA_DIR
    if folder:
        save_dir = os.path.join(ALDATA_DIR, folder)
    os.makedirs(save_dir, ok=True)

    if filetype == "PASCALVOC":
        xml_str = _to_pascalvoc(data)
        filepath = os.path.join(save_dir, filename + ".xml")
        with open(filepath, "w") as f:
            f.write(xml_str)
    else:
        filepath = os.path.join(save_dir, filename + ".json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    rel_path = os.path.relpath(filepath, os.path.dirname(__file__))
    log.info("Data saved: %s", rel_path)
    return jsonify({
        "status": "ok",
        "id": uid,
        "tag": tag,
        "folder": folder,
        "filetype": filetype,
        "path": rel_path,
    })


@app.route("/alldata", methods=["GET"])
def list_data():
    folder = request.args.get("FOLDER", "").strip()
    tag = request.args.get("TAG", "").strip()
    query = request.args.get("query", "").strip()
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)

    search_dir = ALDATA_DIR
    if folder:
        search_dir = os.path.join(ALDATA_DIR, folder)

    if not os.path.isdir(search_dir):
        return jsonify({"files": [], "total": 0, "folder": folder, "tag": tag, "query": query})

    pattern = os.path.join(search_dir, "**", "*.*")
    all_files = glob.glob(pattern, recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f) and (f.endswith(".json") or f.endswith(".xml"))]

    if tag:
        t = tag.lower()
        all_files = [f for f in all_files if t in os.path.basename(f).lower()]

    if query:
        q = query.lower()
        all_files = [f for f in all_files if q in os.path.basename(f).lower()]

    all_files.sort(reverse=True)
    total = len(all_files)
    page = all_files[offset:offset + limit]

    results = []
    for fp in page:
        stat = os.stat(fp)
        results.append({
            "path": os.path.relpath(fp, os.path.dirname(__file__)),
            "name": os.path.basename(fp),
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })

    return jsonify({
        "files": results,
        "total": total,
        "offset": offset,
        "limit": limit,
        "folder": folder,
        "tag": tag,
        "query": query,
    })


@app.route("/alldata/<path:file_id>", methods=["GET"])
def get_data(file_id):
    if not file_id:
        return jsonify({"error": "Missing file ID"}), 400

    candidates = glob.glob(os.path.join(ALDATA_DIR, "**", f"{file_id}.*"), recursive=True)
    if not candidates:
        return jsonify({"error": f"No data file found matching '{file_id}'"}), 404

    filepath = candidates[0]
    if filepath.endswith(".json"):
        with open(filepath) as f:
            data = json.load(f)
        return jsonify({"id": file_id, "path": os.path.relpath(filepath, os.path.dirname(__file__)), "data": data})
    else:
        mime = "text/xml"
        return send_file(filepath, mimetype=mime)


@app.route("/alldata/<path:file_id>", methods=["DELETE"])
def delete_data(file_id):
    if not file_id:
        return jsonify({"error": "Missing file ID"}), 400

    candidates = glob.glob(os.path.join(ALDATA_DIR, "**", f"{file_id}.*"), recursive=True)
    if not candidates:
        return jsonify({"error": f"No data file found matching '{file_id}'"}), 404

    filepath = candidates[0]
    os.remove(filepath)
    log.info("Deleted alldata: %s", os.path.relpath(filepath, os.path.dirname(__file__)))
    return jsonify({"status": "ok", "deleted": os.path.relpath(filepath, os.path.dirname(__file__))})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", config.get("port", 5000)))
    try:
        manager.load(DEFAULT_MODEL)
    except Exception as e:
        log.error("Startup model load failed: %s. Continuing without model.", e)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "0") == "1")
