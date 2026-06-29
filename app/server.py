"""AutoMaster web service — upload audio, get a Boris-mastered WAV back.

FastAPI + a tiny static UI. The heavy DSP is Python (torch/scipy), so this is
built to run on a Python host (e.g. Render) rather than a JS-serverless edge.

    uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io as _io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from automaster import io as amio, metrics, render
from automaster.model import EditorModel

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AutoMaster (BCCM)")

_MODEL_CACHE: dict[str, EditorModel] = {}


def _get_model(editor: str) -> EditorModel:
    if editor not in _MODEL_CACHE:
        path = MODELS / f"{editor}.json"
        if not path.exists():
            raise HTTPException(404, f"no model for editor '{editor}'")
        _MODEL_CACHE[editor] = EditorModel.load(path)
    return _MODEL_CACHE[editor]


@app.get("/api/models")
def list_models():
    out = []
    for p in sorted(MODELS.glob("*.json")):
        m = EditorModel.load(p)
        out.append({"editor": m.editor, "target_lufs": m.target_lufs,
                    "n_pairs": m.n_pairs, "notes": m.notes})
    return {"models": out}


@app.post("/api/master")
async def master(
    file: UploadFile = File(...),
    editor: str = Form("boris"),
    limiter: bool = Form(True),
    ceiling_db: float = Form(-1.0),
):
    model = _get_model(editor)
    suffix = Path(file.filename or "in").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        in_path = tmp.name

    try:
        x, sr = amio.load_audio(in_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode audio: {e}")

    before = metrics.measure(x, sr)
    y = render.render(x, sr, model, apply_limiter=limiter, ceiling_db=ceiling_db)
    after = metrics.measure(y, sr)

    buf = _io.BytesIO()
    sf.write(buf, np.asarray(y), sr, format="WAV", subtype="PCM_24")
    buf.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="mastered_{editor}.wav"',
        "X-Before-LUFS": f"{before['lufs']:.2f}",
        "X-After-LUFS": f"{after['lufs']:.2f}",
        "X-Before-TP": f"{before['tp_db']:.2f}",
        "X-After-TP": f"{after['tp_db']:.2f}",
        "Access-Control-Expose-Headers": "X-Before-LUFS,X-After-LUFS,X-Before-TP,X-After-TP",
    }
    return StreamingResponse(buf, media_type="audio/wav", headers=headers)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
