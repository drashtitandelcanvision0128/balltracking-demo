FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY backend/requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

ENV PYTHONPATH=/app/backend
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
