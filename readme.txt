YOLO OBJECT DETECTION API
=========================
Base URL: http://<host>:<port>  (default port 5000)
All request/response bodies are JSON (Content-Type: application/json) unless noted.
CORS is fully open (Access-Control-Allow-Origin: *).


--- MODEL MANAGEMENT ---

Models are loaded/unloaded dynamically. Up to MAX_LOADED_MODELS (default 5) can
reside in GPU VRAM at once. If a predict request targets an unloaded model, it is
auto-loaded (evicting the least-recently-used model if at capacity).

Built-in models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
Custom models:   place .pt files in models/<name>/weights/ or models/<name>.pt


GET /health
-----------
Server status. No body required.

curl http://localhost:5000/health

Response 200:
{
  "status": "ok",
  "default_model": "yolov8n",        // used when predict omits "model"
  "max_loaded_models": 5,            // from .env MAX_LOADED_MODELS
  "loaded_count": 2,                 // how many models are in GPU
  "loaded_models": ["yolov8n", "yolov8s"]  // names of loaded models
}


GET /models
-----------
List all discoverable models (built-in + custom from models/ dir).

curl http://localhost:5000/models

Response 200:
{
  "models": [
    {"name": "yolov8n", "label": "yolov8n (nano)", "source": "built-in"},
    {"name": "yolov8s", "label": "yolov8s (small)", "source": "built-in"},
    {"name": "yolov8m", "label": "yolov8m (medium)", "source": "built-in"},
    {"name": "yolov8l", "label": "yolov8l (large)", "source": "built-in"},
    {"name": "yolov8x", "label": "yolov8x (xlarge)", "source": "built-in"}
  ],
  "default": "yolov8n"
}
// Custom models from models/ dir have "source": "models/" and a "path" field.


GET /loaded
-----------
List currently loaded models with VRAM usage and GPU info.

curl http://localhost:5000/loaded

Response 200:
{
  "models": [
    {
      "name": "yolov8n",
      "path": "/home/ubuntu/yolo-api/yolov8n.pt",
      "loaded_at": 1716050000.123,     // unix timestamp
      "vram_bytes": 6815744            // measured VRAM consumption (0 if unknown)
    }
  ],
  "count": 1,
  "max_loaded": 5,
  "gpu": {
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "total_vram": 12582912000,
    "allocated_vram": 6815744,
    "reserved_vram": 2097152
  }
}
// "gpu" is empty {} if no CUDA GPU is available.
// vram_bytes per model is measured after load via torch.cuda.memory_allocated delta.


POST /load
----------
Load a model into GPU memory.

curl -X POST http://localhost:5000/load \
  -H "Content-Type: application/json" \
  -d '{"model": "yolov8s"}'

Request body:
{
  "model": "yolov8s"    // required. name (e.g. "yolov8n") or full path to .pt file
}

Response 200 (newly loaded):
{
  "status": "ok",
  "model": "yolov8s",
  "path": "/home/ubuntu/yolo-api/yolov8s.pt",
  "message": "Model 'yolov8s' loaded"
}

Response 200 (already loaded):
{
  "status": "ok",
  "model": "yolov8s",
  "message": "Already loaded"
}

Response 404 (file not found):
{
  "error": "Model file not found: /some/path.pt"
}


POST /unload
------------
Remove a model from GPU, freeing its VRAM.

curl -X POST http://localhost:5000/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "yolov8s"}'

Request body:
{
  "model": "yolov8s"    // required
}

Response 200:
{
  "status": "ok",
  "model": "yolov8s",
  "message": "Model 'yolov8s' unloaded"
}

Response 404 (not loaded):
{
  "error": "Model 'yolov8s' is not loaded"
}


--- PREDICTION ---

POST /predict
-------------
Run object detection on an image. Optionally specify which model to use.
If the model is not loaded, it is auto-loaded (may evict another model).

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'

# With specific model:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-string>", "model": "yolov8x"}'

# From a file (bash one-liner):
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 image.jpg)\"}"

# Pascal VOC XML output (append query param):
curl -X POST "http://localhost:5000/predict?format=pascalvoc" \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-string>"}'

Request body:
{
  "image": "...",    // required. base64-encoded image (JPEG, PNG, etc.)
  "model": "..."     // optional. model name or path. defaults to default_model.
}

Response 200 (JSON, default):
{
  "model": "yolov8n",       // which model performed detection
  "count": 2,               // number of detected objects
  "objects": [
    {
      "class": 0,           // integer class ID (COCO dataset index)
      "label": "person",    // human-readable class name
      "confidence": 0.8658, // detection confidence 0.0-1.0
      "bbox": {
        "x1": 48.55,        // top-left corner X (pixels)
        "y1": 398.56,       // top-left corner Y (pixels)
        "x2": 245.34,       // bottom-right corner X (pixels)
        "y2": 902.71        // bottom-right corner Y (pixels)
      }
    },
    {
      "class": 5,
      "label": "bus",
      "confidence": 0.8733,
      "bbox": { "x1": 22.88, "y1": 231.26, "x2": 805.0, "y2": 756.83 }
    }
  ]
}

Response 200 (Pascal VOC XML, when ?format=pascalvoc):
<?xml version="1.0" ?>
<annotation>
  <filename>image</filename>
  <size>
    <width>800</width>
    <height>600</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <confidence>0.8658</confidence>
    <bndbox>
      <xmin>48</xmin>
      <ymin>398</ymin>
      <xmax>245</xmax>
      <ymax>902</ymax>
    </bndbox>
  </object>
</annotation>

Response 400:
{
  "error": "Missing 'image' field (base64 string)"
}
// or "Invalid base64 data" or "Could not decode image"

Response 500:
{
  "error": "<exception message>"
}


--- WEB UI PAGES (GET, return HTML) ---

GET /gui       - Interactive prediction GUI with image upload and canvas overlay
GET /manage    - Model management dashboard (load/unload, GPU status, VRAM bar)
GET /files     - File manager UI (bookmarks, upload, search, test API calls)
GET /alldata_ui - Data manager UI (browse/save/delete annotation data)
GET /settings  - Redirects to /manage (backwards compatibility)


--- FILE / FOLDER MANAGEMENT ---

Bookmarks map named identifiers to directory paths with glob patterns.
Paths in bookmarks are relative to the project root.
Persistent storage: bookmarks.json (auto-created on first write).


GET /getbookmarks
-----------------
List all bookmarks. No body required.

curl http://localhost:5000/getbookmarks

Response 200:
{
  "bookmarks": {
    "USH": {
      "path": "data/USH",         // relative to project root
      "pattern": "*/*.mp4",       // glob appended to path
      "label": "USH Videos"       // display label
    },
    "PHOTOS": {
      "path": "data/photos",
      "pattern": "**/*.jpg",
      "label": "Photo Archive"
    }
  }
}


POST /bookmarks
---------------
Add or update a bookmark.

curl -X POST http://localhost:5000/bookmarks \
  -H "Content-Type: application/json" \
  -d '{"name":"USH","path":"data/USH","pattern":"*/*.mp4","label":"USH Videos"}'

Request body:
{
  "name": "USH",          // required. unique identifier
  "path": "data/USH",    // required. directory path (relative to project root)
  "pattern": "*/*.mp4",  // optional. glob pattern. default "**/*"
  "label": "USH Videos"  // optional. display label. defaults to name
}

Response 200:
{
  "status": "ok",
  "name": "USH"
}


DELETE /bookmarks
-----------------
Remove a bookmark.

curl -X DELETE http://localhost:5000/bookmarks \
  -H "Content-Type: application/json" \
  -d '{"name":"USH"}'

Request body:
{
  "name": "USH"    // required
}

Response 200:
{
  "status": "ok",
  "name": "USH"
}

Response 404:
{
  "error": "Bookmark 'USH' not found"
}


GET /getfile?bookmark=USH&query=2026&limit=1
--------------------------------------------
Search for files within a bookmark's directory.

Parameters (query string):
  bookmark  (required) - bookmark name
  query     (optional) - substring filter matched against filenames (case-insensitive)
  limit     (optional) - max results. default 1.

curl "http://localhost:5000/getfile?bookmark=USH&query=2026&limit=1"

When limit=1 and file found: serves the actual file bytes with correct Content-Type.
  Example: video/mp4 for .mp4 files, image/jpeg for .jpg files.
  The browser/player can stream or display it directly.

When limit>1 and files found: returns JSON with file metadata:
{
  "bookmark": "USH",
  "query": "2026",
  "count": 3,
  "files": [
    {
      "path": "data/USH/2026/video1.mp4",   // relative to project root
      "name": "video1.mp4",
      "size": 104857600,                     // bytes
      "mime": "video/mp4"
    }
  ]
}

When NOT found (no match or bookmark missing): returns JSON with sample fallback:
{
  "status": "notfound",
  "bookmark": "USH",
  "query": "nonexistent.mp4",
  "sample": "sample/sample.mp4",    // path to placeholder file
  "ext": ".mp4"                     // extension that was searched for
}
// The "sample" field lets the frontend show a placeholder.
// ext is .mp4 for video or .jpg/.jpeg/.png for images (auto-detected from query).

Response 400:
{
  "error": "Missing 'bookmark' parameter"
}


GET /filecounts?bookmark=USH&query=2026
----------------------------------------
Count files in a bookmark area. Optional query filter.

Parameters (query string):
  bookmark  (required) - bookmark name
  query     (optional) - substring filter (case-insensitive)

curl "http://localhost:5000/filecounts?bookmark=USH&query=2026"

Response 200:
{
  "bookmark": "USH",
  "total": 142,        // total files matching the glob pattern
  "filtered": 23,      // files matching both glob AND query substring
  "query": "2026"      // the query that was applied (empty string if none)
}
// If bookmark doesn't exist: total=0, filtered=0


--- DATA SAVE / RETRIEVE (ALDATA) ---

Save and retrieve annotation data (Pascal VOC XML or JSON).
Files are stored in alldata/{FOLDER}/{unique_id}_{TAG}.xml|.json
If no FOLDER is given, files save directly to alldata/.

File naming: {12-char-hex-id}_{TAG}.{ext}
  Example: a1b2c3d4e5f6_UNICORNS.json


POST /alldata?TAG=UNICORNS&FOLDER=HORSES&filetype=PACLVOC
----------------------------------------------------------
Save annotation data. Body is raw JSON.

Parameters (query string):
  TAG       (required) - label appended to filename
  FOLDER    (optional) - subfolder inside alldata/
  filetype  (optional) - "JSON" (default) or "PASCALVOC"

# Save as JSON:
curl -X POST "http://localhost:5000/alldata?TAG=UNICORNS&FOLDER=HORSES" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "image001.jpg",
    "folder": "HORSES",
    "size": {"width": 800, "height": 600, "depth": 3},
    "objects": [
      {
        "name": "horse",
        "confidence": 0.95,
        "bbox": {"x1": 100, "y1": 50, "x2": 400, "y2": 350}
      }
    ]
  }'

# Save as Pascal VOC XML:
curl -X POST "http://localhost:5000/alldata?TAG=UNICORNS&FOLDER=HORSES&filetype=PASCALVOC" \
  -H "Content-Type: application/json" \
  -d '{"filename":"image001.jpg","objects":[{"name":"horse","bbox":{"x1":100,"y1":50,"x2":400,"y2":350}}]}'

Request body: raw JSON. Any structure is accepted.
  For PASCALVOC, these keys are mapped to XML tags:
    filename, folder, size (width/height/depth), source, objects/object
    Each object: name, confidence, pose, truncated, difficult, bndbox/bbox
  All other top-level keys become child elements of <annotation>.
  For JSON mode, the body is saved as-is (pretty-printed).

Response 200:
{
  "status": "ok",
  "id": "a1b2c3d4e5f6",      // 12-char hex ID (use to retrieve later)
  "tag": "UNICORNS",
  "folder": "HORSES",
  "filetype": "JSON",          // or "PASCALVOC"
  "path": "alldata/HORSES/a1b2c3d4e5f6_UNICORNS.json"  // relative to project root
}

Response 400:
{
  "error": "Missing ?TAG= parameter"
}


GET /alldata?FOLDER=HORSES&TAG=UNICORNS&query=2026&limit=50&offset=0
-------------------------------------------------------------------
List saved data files. All parameters are optional.

Parameters (query string):
  FOLDER  (optional) - filter by subfolder
  TAG     (optional) - filter by tag in filename
  query   (optional) - substring filter on filename
  limit   (optional) - max results per page. default 50.
  offset  (optional) - pagination offset. default 0.

curl "http://localhost:5000/alldata?FOLDER=HORSES&TAG=UNICORNS"

Response 200:
{
  "files": [
    {
      "path": "alldata/HORSES/a1b2c3d4e5f6_UNICORNS.json",
      "name": "a1b2c3d4e5f6_UNICORNS.json",
      "size": 1024,              // bytes
      "modified": 1716050000.123 // unix timestamp
    }
  ],
  "total": 42,        // total matching files (before pagination)
  "offset": 0,
  "limit": 50,
  "folder": "HORSES",
  "tag": "UNICORNS",
  "query": ""
}


GET /alldata/{id}
-----------------
Retrieve a saved data file by its 12-char hex ID.

curl http://localhost:5000/alldata/a1b2c3d4e5f6

# Or with full path:
curl http://localhost:5000/alldata/HORSES/a1b2c3d4e5f6_UNICORNS

For .json files, returns:
{
  "id": "a1b2c3d4e5f6",
  "path": "alldata/HORSES/a1b2c3d4e5f6_UNICORNS.json",
  "data": { ... }              // the original JSON body that was saved
}

For .xml files, returns the raw XML with Content-Type: text/xml.

The ID search is recursive — it finds files matching {id}.* anywhere
under alldata/. So "a1b2c3d4e5f6" matches "alldata/HORSES/a1b2c3d4e5f6_UNICORNS.json".

Response 404:
{
  "error": "No data file found matching 'a1b2c3d4e5f6'"
}


--- CONFIGURATION ---

.env:
  MAX_LOADED_MODELS=5    Max models in GPU simultaneously. Adjust as needed.

config.json:
  {"default_model": "yolov8n"}   Fallback model when predict omits "model".

Models resolve in this order:
  1. Exact file path (e.g. "/path/to/my.pt")
  2. Name + ".pt" in project root (e.g. "yolov8n" -> "yolov8n.pt")
  3. Custom model in models/ directory
  4. Used as-is (will fail with FileNotFoundError if invalid)
