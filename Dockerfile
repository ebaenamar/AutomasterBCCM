# AutoMaster (BCCM) — container with ffmpeg so .mp4/.m4a decode too.
FROM python:3.12-slim

# ffmpeg/ffprobe for video & compressed audio decode. libsndfile ships inside
# the soundfile wheel, so no extra system audio libs are needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching (CPU torch).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs, then install the package without
# re-resolving deps (already satisfied above; keeps the image lean — no
# pandas/matplotlib, which are analysis-only).
COPY pyproject.toml ./
COPY src ./src
COPY app ./app
COPY models ./models
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
