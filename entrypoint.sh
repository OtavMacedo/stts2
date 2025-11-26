#!/bin/bash
set -e

python ./download_models.py || exit 1

exec uvicorn src.main:app --workers 1 --host 0.0.0.0 --port 8000