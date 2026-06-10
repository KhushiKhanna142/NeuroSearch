"""
FastAPI backend for the NeuroSearch dashboard.
Serves search data, architecture specs, and runs ONNX inference.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.log_parser import parse_log, get_status
from backend.inference import run_inference

app = FastAPI(title="NeuroSearch API", version="1.0")

# Allow React dev server on port 5173 / 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = Path(__file__).parent.parent

# Prefer full-run, fall back to smoke-test exports
def _results_path():
    full = _BASE / 'exports' / 'search_results.json'
    smoke = _BASE / 'exports' / 'smoke' / 'search_results.json'
    if full.exists():
        return full
    if smoke.exists():
        return smoke
    return None

# Load arch pool from search checkpoint if available
def _load_arch_pool():
    import torch
    ckpt_paths = [
        _BASE / 'checkpoints' / 'search.pt',
        _BASE / 'checkpoints' / 'smoke_search.pt',
    ]
    for p in ckpt_paths:
        if p.exists():
            try:
                ckpt = torch.load(p, map_location='cpu')
                return ckpt.get('arch_pool', [])
            except Exception:
                pass
    return []


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/api/status")
def status():
    """Current pipeline phase: idle / pretraining / searching / complete"""
    phase = get_status()
    return {"status": phase}


@app.get("/api/search-results")
def search_results():
    """Top-5 architectures with accuracy, FLOPs, reward."""
    path = _results_path()
    if path is None:
        raise HTTPException(404, detail="search_results.json not found — run the pipeline first")
    with open(path) as f:
        data = json.load(f)
    return data


@app.get("/api/search-log")
def search_log():
    """Full episode log as structured JSON array (from log file or checkpoint)."""
    _, episodes = parse_log()

    # If no log file, generate synthetic data from the checkpoint pool
    if not episodes:
        pool = _load_arch_pool()
        episodes = [
            {
                'episode':    e.get('episode', i + 1),
                'accuracy':   e.get('accuracy', 0),
                'flops':      e.get('flops', 0),
                'raw_reward': e.get('reward', 0),
                'reward':     e.get('reward', 0),
                'entropy':    2.079,
                'is_new':     True,
            }
            for i, e in enumerate(pool)
        ]

    return {"episodes": episodes}


@app.get("/api/architecture/{rank}")
def architecture(rank: int):
    """
    Return the arch_spec for the rank-N architecture (1-indexed).
    arch_spec is a list of dicts: edge_key -> op_name per cell.
    """
    pool = _load_arch_pool()
    if not pool:
        # Fall back to search_results.json which has top-5 basic info
        path = _results_path()
        if path is None:
            raise HTTPException(404, "No architecture data found")
        with open(path) as f:
            data = json.load(f)
        top5 = data.get('top5', [])
        if rank < 1 or rank > len(top5):
            raise HTTPException(404, f"Rank {rank} not found (only {len(top5)} available)")
        return top5[rank - 1]

    # Sort by reward descending
    sorted_pool = sorted(pool, key=lambda e: e.get('reward', 0), reverse=True)
    if rank < 1 or rank > len(sorted_pool):
        raise HTTPException(404, f"Rank {rank} not found")

    entry = sorted_pool[rank - 1]
    return {
        'rank':      rank,
        'accuracy':  entry.get('accuracy', 0),
        'flops':     entry.get('flops', 0),
        'reward':    entry.get('reward', 0),
        'episode':   entry.get('episode'),
        'arch_spec': entry.get('arch', []),
    }


@app.post("/api/inference")
async def inference(file: UploadFile = File(...)):
    """
    Accept an image upload and return top-5 class predictions.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image (PNG or JPG)")
    image_bytes = await file.read()
    try:
        result = run_inference(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    return result


@app.get("/api/finetune-results")
def finetune_results():
    """Return the final post-search fine-tuning results if available."""
    path = _BASE / 'exports' / 'finetune_results.json'
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data


@app.get("/api/stream-log")
async def stream_log():
    """
    SSE stream of live log lines.
    Tails logs/train_log.txt and sends new lines as server-sent events.
    """
    log_path = _BASE / 'logs' / 'train_log.txt'

    async def event_generator():
        last_pos = 0
        while True:
            if log_path.exists():
                with open(log_path) as f:
                    f.seek(last_pos)
                    lines = f.readlines()
                    last_pos = f.tell()
                for line in lines:
                    yield f"data: {line.strip()}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serve built static files from frontend/dist if they exist
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

dist_path = _BASE / 'frontend' / 'dist'
if dist_path.exists():
    app.mount("/assets", StaticFiles(directory=dist_path / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            raise HTTPException(404)
        return FileResponse(dist_path / "index.html")
