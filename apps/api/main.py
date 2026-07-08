"""Memo-House backend.

Accepts photo uploads, turns each into a Gaussian-splat PLY using the SHARP
model, stores the result, and exposes the latest scene to the viewer.

Run via:  uvicorn main:app   (with the ml-sharp venv active)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import proxy as proxy_mod
from sharp_engine import SharpEngine
from storage import Scene, Storage, now

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("memo-haus.api")

# --- configuration -----------------------------------------------------
_DEFAULT_STORAGE = Path(__file__).resolve().parent.parent.parent / "storage"
STORAGE_DIR = Path(os.environ.get("MEMO_STORAGE_DIR", _DEFAULT_STORAGE)).resolve()
DEVICE = os.environ.get("MEMO_DEVICE", "default")
CHECKPOINT = os.environ.get("MEMO_CHECKPOINT")
WARMUP_TIMEOUT = float(os.environ.get("MEMO_WARMUP_TIMEOUT", "900"))

# Populated lazily on first upload (deferred to avoid importing torch at startup)
SUPPORTED_EXTENSIONS: set[str] = set()

storage = Storage(STORAGE_DIR)
engine = SharpEngine(device=DEVICE, checkpoint_path=Path(CHECKPOINT) if CHECKPOINT else None)

_ready = threading.Event()        # set once model is loaded
_warmup_started = threading.Event()  # set once the load thread has been kicked off
_processing = threading.Event()   # set while a prediction is running

# Lets the mobile app tell the viewer "show this specific memory now" without
# touching the viewer's own polling/rotation logic at all — the viewer just
# additionally checks this and calls its existing goToIndex() when it changes.
_selection_lock = threading.Lock()
_selected_scene_id: str | None = None
_selected_at: float = 0.0

# Same idea, but for Memory Verse: a curated list of ids instead of one, so
# the world only ever loads scenes the visitor actually picked — keeps it
# fast and avoids placing dozens of scenes (and giant stitched merges) at
# once, which is what made it laggy and prone to clashing.
_world_selected_ids: list[str] = []
_world_selected_at: float = 0.0

_nav_lock = threading.Lock()
_nav_state: dict = {"move_x":0,"move_z":0,"move_y":0,"turn_x":0,"turn_y":0,"gyro":False,"gyro_yaw":None,"gyro_pitch":None,"ts":0}
_reset_ts: float = 0.0

# Where each memory sits on the shared ground-plane map, as fractions of the
# unit square (0..1). This is the single source of truth for placement: the
# main-page map edits it, and the 3D viewer maps it onto the ground plane so
# the two always agree. Persisted to disk so a memory's spot survives restarts.
_POS_MIN_DIST = 0.12          # keep auto-placed memories at least this far apart
_positions_lock = threading.Lock()
_positions_path = STORAGE_DIR / "scene_positions.json"


def _load_positions() -> dict:
    try:
        return json.loads(_positions_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_positions() -> None:
    try:
        _positions_path.write_text(json.dumps(_scene_positions), encoding="utf-8")
    except Exception:  # pragma: no cover
        LOGGER.exception("Failed persisting scene positions")


_scene_positions: dict = _load_positions()  # scene_id → {x_pct, y_pct}


def _assign_free_position(scene_id: str) -> dict:
    """Give a memory the first free spot on the map that doesn't collide with an
    existing one — a golden-angle spiral outward from the centre, so new
    memories fan out evenly and never land on top of another."""
    with _positions_lock:
        if scene_id in _scene_positions:
            return _scene_positions[scene_id]
        existing = list(_scene_positions.values())

        def free(x: float, y: float) -> bool:
            return all((x - p["x_pct"]) ** 2 + (y - p["y_pct"]) ** 2 >= _POS_MIN_DIST ** 2
                       for p in existing)

        ga = math.pi * (3 - math.sqrt(5))
        chosen = None
        for i in range(600):
            r = 0.06 + 0.03 * math.sqrt(i)
            x = min(0.96, max(0.04, 0.5 + r * math.cos(i * ga)))
            y = min(0.96, max(0.04, 0.5 + r * math.sin(i * ga)))
            if free(x, y):
                chosen = {"x_pct": x, "y_pct": y}
                break
        if chosen is None:
            chosen = {"x_pct": random.random(), "y_pct": random.random()}
        _scene_positions[scene_id] = chosen
        _save_positions()
        return chosen


# Latest 3D viewer camera pose, so the map can draw where the viewer is looking.
_camera_lock = threading.Lock()
_camera_state: dict = {"x": 0.0, "z": 0.0, "yaw": 0.0, "ts": 0.0}


class SelectScenePayload(BaseModel):
    scene_id: str


class WorldSelectionPayload(BaseModel):
    scene_ids: list[str]


class ScenePositionPayload(BaseModel):
    scene_id: str
    x_pct: float
    y_pct: float


class CameraStatePayload(BaseModel):
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


class NavPayload(BaseModel):
    move_x: float = 0
    move_z: float = 0
    move_y: float = 0
    turn_x: float = 0
    turn_y: float = 0
    gyro: bool = False
    gyro_yaw: float | None = None
    gyro_pitch: float | None = None
    ts: float = 0

app = FastAPI(title="Memo-House API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scene assets (splats, source photos, audio) are written once under a
# unique hashed scene id and never modified in place — only ever deleted —
# so it's safe to tell browsers to cache them forever instead of
# re-downloading the same multi-megabyte PLY on every viewer refresh.
_IMMUTABLE_STATIC_PREFIXES = ("/outputs/", "/uploads/", "/audio/")


@app.middleware("http")
async def cache_immutable_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(_IMMUTABLE_STATIC_PREFIXES) and response.status_code == 200:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


def _warmup() -> None:
    try:
        engine.load()
        _ready.set()
    except Exception:  # pragma: no cover - surfaced via /api/health
        LOGGER.exception("Failed to load SHARP model during warmup")


def _ensure_warmup_started() -> None:
    """Kick off the (slow) SHARP load on first use instead of at server
    startup, so `npm run dev` comes up instantly and idle dev sessions don't
    pay the checkpoint load / VRAM cost."""
    if _ready.is_set() or _warmup_started.is_set():
        return
    _warmup_started.set()
    LOGGER.info("Warming up SHARP model in the background (device=%s)...", engine.device)
    threading.Thread(target=_warmup, name="sharp-warmup", daemon=True).start()


def _backfill_positions() -> None:
    """Give every existing scene a map position if it doesn't have one yet, so
    every memory has a spot the moment the viewer/map opens. (Proxies are now
    built fresh per request and aren't cached, so there's nothing to pre-warm.)"""
    try:
        for scene in storage.list_scenes():
            if scene.id not in _scene_positions:
                _assign_free_position(scene.id)
    except Exception:
        LOGGER.exception("Position backfill pass failed")


@app.on_event("startup")
def on_startup() -> None:
    LOGGER.info("Storage at %s", STORAGE_DIR)
    threading.Thread(target=_backfill_positions, name="position-backfill", daemon=True).start()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_ready": engine.ready, "device": engine.device, "processing": _processing.is_set()}


@app.get("/api/latest")
def latest() -> JSONResponse:
    scene = storage.latest()
    return JSONResponse(_serialize(scene) if scene else None)


@app.get("/api/scenes")
def scenes() -> list[dict]:
    return [_serialize(s) for s in storage.list_scenes()]


@app.get("/api/scene-proxy/{scene_id}")
def scene_proxy(scene_id: str) -> Response:
    """Tiny decimated point cloud (~180 KB) for the viewer's distant-view
    layer, so it never has to download the full ~64 MB PLY just to draw a few
    thousand preview points. Built fresh in-memory each request (a few tens of
    ms, see proxy.py) and NOT cached — not on disk, not in the browser — so
    changes to the decimation/colour code always take effect on reload."""
    ply_path = storage.splats_dir / f"{scene_id}.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="scene not found")
    try:
        data = proxy_mod.build_proxy_bytes(ply_path)
    except Exception as exc:  # pragma: no cover - surfaced to the client
        LOGGER.exception("Failed building proxy for %s", scene_id)
        raise HTTPException(status_code=500, detail=f"proxy build failed: {exc}")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/scene-splat/{scene_id}")
def scene_splat(scene_id: str, keep: float = 0.4) -> FileResponse:
    """Decimated full-res gaussian splat (keep ≈ fraction of the ~1.17 M
    gaussians retained) so the viewer's on-approach point-cloud → splat swap
    lands in ~a second instead of ~10 s. keep=1.0 serves the original PLY.
    Built once per keep level and cached to disk; see proxy.build_decimated_ply."""
    ply_path = storage.splats_dir / f"{scene_id}.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="scene not found")
    try:
        out = proxy_mod.get_or_build_lite_ply(ply_path, keep)
    except Exception as exc:  # pragma: no cover - surfaced to the client
        LOGGER.exception("Failed building lite splat for %s", scene_id)
        raise HTTPException(status_code=500, detail=f"splat build failed: {exc}")
    return FileResponse(
        out,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.post("/api/select-scene")
def select_scene(payload: SelectScenePayload) -> dict:
    """Mobile app calls this when the user picks a memory to view. The
    viewer polls GET /api/select-scene and jumps to it via its normal,
    unchanged goToIndex() — this only decides *which* scene, not how the
    viewer shows it."""
    global _selected_scene_id, _selected_at
    with _selection_lock:
        _selected_scene_id = payload.scene_id
        _selected_at = now()
        return {"scene_id": _selected_scene_id, "selected_at": _selected_at}


@app.get("/api/select-scene")
def get_selected_scene() -> dict:
    with _selection_lock:
        return {"scene_id": _selected_scene_id, "selected_at": _selected_at}


@app.post("/api/world-selection")
def set_world_selection(payload: WorldSelectionPayload) -> dict:
    """Mobile app calls this after multi-selecting memories to place in
    Memory Verse. Mirrors /api/select-scene's pattern exactly, just for a
    list instead of one id — the viewer polls GET and enters the world with
    exactly this set, leaving everything else about it unchanged."""
    global _world_selected_ids, _world_selected_at
    with _selection_lock:
        _world_selected_ids = payload.scene_ids
        _world_selected_at = now()
        return {"scene_ids": _world_selected_ids, "selected_at": _world_selected_at}


@app.get("/api/world-selection")
def get_world_selection() -> dict:
    with _selection_lock:
        return {"scene_ids": _world_selected_ids, "selected_at": _world_selected_at}


@app.post("/api/navigate")
def set_navigate(payload: NavPayload) -> dict:
    global _nav_state
    data = payload.model_dump()
    data["ts"] = time.time() * 1000  # server-side timestamp avoids mobile/desktop clock skew
    with _nav_lock:
        _nav_state = data
    return {"ok": True}


@app.get("/api/navigate")
def get_navigate() -> dict:
    with _nav_lock:
        return dict(_nav_state)


@app.post("/api/scene-position")
def set_scene_position(payload: ScenePositionPayload) -> dict:
    with _positions_lock:
        _scene_positions[payload.scene_id] = {"x_pct": payload.x_pct, "y_pct": payload.y_pct}
        _save_positions()
    return {"ok": True}


@app.get("/api/scene-positions")
def get_scene_positions() -> dict:
    with _positions_lock:
        return dict(_scene_positions)


@app.post("/api/camera-state")
def set_camera_state(payload: CameraStatePayload) -> dict:
    """The 3D viewer posts its camera pose here a few times a second so the
    main-page map can show where the viewer is and which way it's looking."""
    global _camera_state
    with _camera_lock:
        _camera_state = {"x": payload.x, "z": payload.z, "yaw": payload.yaw, "ts": time.time() * 1000}
    return {"ok": True}


@app.get("/api/camera-state")
def get_camera_state() -> dict:
    with _camera_lock:
        return dict(_camera_state)


@app.post("/api/reset-view")
def post_reset_view() -> dict:
    global _reset_ts
    _reset_ts = time.time() * 1000
    return {"ok": True, "ts": _reset_ts}


@app.get("/api/reset-view")
def get_reset_view() -> dict:
    return {"ts": _reset_ts}


@app.delete("/api/scenes/{scene_id}")
def delete_scene(scene_id: str) -> dict:
    scene = storage.delete_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    LOGGER.info("Deleted scene %s (and its files)", scene_id)
    return {"deleted": scene_id}


@app.get("/api/music/search")
def music_search(q: str) -> list[dict]:
    """Proxies Openverse's free, keyless audio search (Creative Commons /
    openly-licensed tracks) — the mobile app's "Browse Music" picker calls
    this directly, no API key needed on either side."""
    if not q.strip():
        return []
    try:
        resp = requests.get(
            "https://api.openverse.org/v1/audio/",
            params={"q": q, "page_size": 12},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Music search failed: {exc}") from exc

    return [
        {
            "id": item.get("id"),
            "title": item.get("title") or "Untitled",
            "creator": item.get("creator") or "Unknown artist",
            "audio_url": item.get("url"),
            "duration_ms": item.get("duration"),
            "license": item.get("license"),
        }
        for item in data.get("results", [])
        if item.get("url")
    ]


@app.get("/api/music/fetch")
def music_fetch(url: str) -> StreamingResponse:
    """Streams a third-party track through our own origin. The mobile app's
    crop UI needs to decode the audio with the Web Audio API, which fails on
    cross-origin media unless the third-party host sends CORS headers (most
    don't) — fetching same-origin through here sidesteps that entirely."""
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid url")
    try:
        upstream = requests.get(url, stream=True, timeout=15)
        upstream.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch track: {exc}") from exc
    content_type = upstream.headers.get("Content-Type", "audio/mpeg")
    return StreamingResponse(upstream.iter_content(chunk_size=8192), media_type=content_type)


@app.post("/api/predict")
def predict(
    image: UploadFile = File(...),
    name: str = Form(""),
    author: str = Form(""),
    year: str = Form(""),
    story: str = Form(""),
    audio: UploadFile | None = File(None),
) -> dict:
    # Populate supported extensions on first upload (torch must be loaded by then)
    if not SUPPORTED_EXTENSIONS:
        from sharp.utils import io as sharp_io
        SUPPORTED_EXTENSIONS.update(ext.lower() for ext in sharp_io.get_supported_image_extensions())

    suffix = Path(image.filename or "").suffix.lower()

    # Some mobile browsers (especially Android) don't include an extension in
    # the uploaded filename. Fall back to guessing from the MIME type.
    if not suffix and image.content_type:
        import mimetypes
        guessed = mimetypes.guess_extension(image.content_type)
        # guess_extension returns things like '.jpe' / '.jpeg' — normalise to common forms
        _mime_map = {
            ".jpe": ".jpg", ".jpeg": ".jpg", ".jfif": ".jpg",
            ".tiff": ".tif", ".heic": ".heic", ".heif": ".heif",
        }
        if guessed:
            suffix = _mime_map.get(guessed, guessed)
        LOGGER.info("No extension in filename; inferred '%s' from content-type '%s'", suffix, image.content_type)

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{suffix or image.content_type or image.filename}'. "
                   f"Please upload a JPEG, PNG, or WebP photo.",
        )

    # Block until the model is loaded (first run may download the checkpoint).
    _ensure_warmup_started()
    if not _ready.wait(timeout=WARMUP_TIMEOUT):
        raise HTTPException(status_code=503, detail="Model is still warming up. Try again shortly.")

    scene_id, upload_path, ply_path = storage.new_scene_paths(
        suffix, name=name, year=year, story=story
    )

    with upload_path.open("wb") as f:
        f.write(image.file.read())

    # Save the optional voice note / cropped music clip under the same
    # filename stem as the photo and splat — same "tagging" scheme as the
    # image, just in Memo-audio instead of Memo-album.
    audio_filename = ""
    if audio is not None and audio.filename:
        audio_suffix = Path(audio.filename).suffix.lower() or ".webm"
        audio_path = storage.audio_path_for(scene_id, audio_suffix)
        with audio_path.open("wb") as f:
            f.write(audio.file.read())
        audio_filename = audio_path.name
        LOGGER.info("Saved audio for scene %s -> %s", scene_id, audio_filename)

    LOGGER.info("Running SHARP inference for scene %s (%s)", scene_id, upload_path.name)
    _processing.set()
    try:
        engine.predict_to_ply(upload_path, ply_path)
    except Exception as exc:
        LOGGER.exception("Inference failed for scene %s", scene_id)
        msg = str(exc)
        if "out of memory" in msg.lower() or "cuda error" in msg.lower():
            detail = (
                "GPU out of memory. Please restart the API server — "
                "the updated code loads the model in fp16 (half memory) and "
                "automatically falls back to CPU if needed."
            )
        else:
            detail = f"Inference failed: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        _processing.clear()

    scene = storage.add_scene(
        Scene(
            id=scene_id,
            name=name.strip() or "Untitled",
            author=author.strip() or "Anonymous",
            ply_file=ply_path.name,
            image_file=upload_path.name,
            created_at=now(),
            year=year.strip(),
            story=story.strip(),
            cluster_id="",
            audio_file=audio_filename,
        )
    )
    # Give the new memory a spot on the map that doesn't overlap any existing
    # one, so it fans out into free space instead of landing on a neighbour.
    _assign_free_position(scene_id)
    LOGGER.info("Scene %s ready -> %s", scene_id, scene.ply_url)
    return _serialize(scene)


def _serialize(scene: Scene) -> dict:
    return {
        "id": scene.id,
        "name": scene.name,
        "author": scene.author,
        "year": scene.year,
        "story": scene.story,
        "cluster_id": scene.cluster_id,
        "ply_url": scene.ply_url,
        "splat_url": f"/api/scene-splat/{scene.id}",
        "image_url": scene.image_url,
        "audio_url": scene.audio_url,
        "created_at": scene.created_at,
    }


app.mount("/outputs", StaticFiles(directory=str(storage.splats_dir)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(storage.uploads_dir)), name="uploads")
app.mount("/audio", StaticFiles(directory=str(storage.audio_dir)), name="audio")
