FROM python:3.12-slim

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    build-essential \
    python3-dev \
    libopenblas-dev \
    libsndfile1-dev \
    ffmpeg \
    espeak \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--workers", "1", "--host", "0.0.0.0", "--port", "8000"]
