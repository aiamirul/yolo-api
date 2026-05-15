import os
import sys
import json
import glob
import base64
import subprocess
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


config = load_config()
MODEL_NAME = config.get("model") or os.environ.get("YOLO_MODEL", "yolov8n")
model = None


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


def resolve_model(name):
    if os.path.isfile(name):
        return name
    if os.path.isfile(f"{name}.pt"):
        return f"{name}.pt"
    for m in scan_models():
        if m["name"] == name and "path" in m:
            return m["path"]
    return name


def get_model():
    global model
    if model is None:
        model = YOLO(resolve_model(MODEL_NAME))
    return model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": scan_models(), "current": MODEL_NAME})


@app.route("/gui", methods=["GET"])
def gui():
    return render_template("gui.html")


@app.route("/settings", methods=["GET"])
def settings_page():
    return render_template("settings.html")


@app.route("/settings", methods=["POST"])
def update_settings():
    global model, MODEL_NAME
    data = request.get_json(force=True)
    new_model = data.get("model", "").strip()
    if not new_model:
        return jsonify({"error": "Missing 'model' field"}), 400

    config["model"] = resolve_model(new_model)
    save_config(config)

    script_dir = os.path.dirname(__file__)
    subprocess.Popen(
        [os.path.join(script_dir, "restart.sh")],
        cwd=script_dir,
        start_new_session=True,
    )

    return jsonify({"status": "ok", "model": new_model, "message": "Server restarting with " + new_model + "..."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Missing 'image' field (base64 string)"}), 400

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 data"}), 400

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        yolo = get_model()
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

        return jsonify({
            "model": MODEL_NAME,
            "count": len(objects),
            "objects": objects,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", config.get("port", 5000)))
    get_model()
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "0") == "1")
