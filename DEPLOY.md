# Deploying AutoMaster

The model is tiny (a JSON preset); all the cost is the DSP, which is CPU-bound
Python (torch + scipy). So deploy it as a **Python web service**, not on a
JS-serverless edge.

## Local with Docker (easiest — includes ffmpeg, so .mp4 works)

```bash
docker compose up --build      # first build ~5-10 min (downloads CPU torch)
# open http://localhost:8000
docker compose down            # stop
```

The image bundles `ffmpeg`, so `.mp4`/`.m4a` uploads decode locally (unlike the
bare Render Python runtime). `models/` is mounted read-only, so a host-side
`python scripts/train_boris.py` is picked up on the next `docker compose up`
without rebuilding. Image is ~2.1 GB (torch); needs ~2 GB RAM for full-length
clips, less for short ones.

Smoke-test it:

```bash
curl -s -F "file=@your.mp4" -F editor=boris http://localhost:8000/api/master -o mastered.wav
```

## Local without Docker

```bash
uv pip install --python .venv -e . fastapi "uvicorn[standard]" python-multipart
./.venv/bin/uvicorn app.server:app --reload --port 8000
# open http://localhost:8000   (.mp4 needs ffmpeg on your PATH)
```

CLI (no server):

```bash
automaster master input.wav -o mastered.wav --editor boris
automaster master input.wav -o hot.wav --editor boris --no-limiter   # replicate mode
```

## Render (recommended)

`render.yaml` is a ready blueprint. Push this repo to GitHub, then on Render:
**New → Blueprint → pick the repo**. It builds with `requirements.txt` and serves
`uvicorn app.server:app`. Use `starter` for short clips; bump RAM for
full-length tracks. The health check hits `/api/models`.

**ffmpeg note:** decoding `.mp4`/`.m4a` needs the `ffmpeg` binary, which the
default Render Python runtime does not include. WAV/FLAC/OGG/MP3 work out of the
box (libsndfile, bundled in the `soundfile` wheel). For mp4 support on Render,
set the service runtime to **Docker** — the repo's `Dockerfile` already installs
ffmpeg, so Render will build and serve that instead of the Python buildpack.

## Vercel

Vercel is great for a **static frontend** but its serverless functions are not a
good fit for multi-hundred-MB torch + multi-second audio jobs (cold starts,
bundle size, execution time limits). If you want Vercel: host `app/static/` on
Vercel and point it at a Render-hosted `/api` (set `const api = "https://<your-render-app>"`
in `index.html` and the CORS header is already exposed).

## The model

`models/boris.json` is produced by `python scripts/train_boris.py` from the
before/after pairs. It stores Boris's target loudness and median EQ/compression;
`gain` is solved per input at render time to hit the target. Retrain whenever new
pairs are added; drop in Kim's model as `models/kim.json` and it appears in the
UI automatically.
