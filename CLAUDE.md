# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Python Flask API that wraps Ultralytics YOLO for object detection with dynamic GPU model management. Models are loaded/unloaded on demand with LRU eviction based on configurable capacity (`MAX_LOADED_MODELS` in `.env`, default 5). The API also includes a bookmark-based file manager and an annotation data store (Pascal VOC XML and JSON).

## Running

```bash
./setup.sh          # creates .venv, installs flask + ultralytics
./start.sh          # starts app.py in foreground (port 5000)
./restart.sh        # kills existing process, starts in background (nohup)
./test_curl.sh      # smoke test: health check + predict on test_bus.jpg
```

The server reads `config.json` for `default_model` (fallback: `yolov8n`). Models resolve in order: exact path → `{name}.pt` in root → `models/` directory → used as-is.

## Architecture

**`app.py`** — single-file Flask application. All routes live here. Key route groups:
- `/health`, `/models`, `/loaded`, `/load`, `/unload` — model lifecycle
- `/predict` — accepts base64 image, returns JSON (default) or Pascal VOC XML (`?format=pascalvoc`)
- `/getbookmarks`, `/bookmarks`, `/getfile`, `/filecounts`, `/upload` — bookmark-based file browsing
- `/alldata`, `/alldata/{id}`, `/alldata_folders` — annotation data CRUD (JSON or Pascal VOC XML)
- `/gui`, `/manage`, `/files`, `/alldata_ui` — web UI pages served from `templates/`

**`model_manager.py`** — `ModelManager` class handles GPU model lifecycle. Thread-safe (uses `threading.Lock`). LRU eviction via `OrderedDict`. Tracks per-model VRAM usage via `torch.cuda.memory_allocated` deltas, persisted to `vram_usage.json`.

**`templates/`** — Jinja2 HTML pages for the web UIs (gui.html, manage.html, files.html, alldata.html).

## Key Design Decisions

- The predict endpoint auto-loads models on demand and evicts LRU models when at capacity, so callers never need to manage model state explicitly.
- Bookmarks map named identifiers to directory paths + glob patterns. Paths in `bookmarks.json` are relative to the project root.
- Annotation data files are named `{12-char-hex-id}_{TAG}.json|xml` and stored under `alldata/`.
- CORS is fully open (`Access-Control-Allow-Origin: *`).
- `config.json` uses the key `"model"` (not `"default_model"`), but `app.py` reads both with a fallback.

## Dependencies

`flask` and `ultralytics` (which pulls in `torch`, `opencv-python`, `numpy`). No test framework is configured.
