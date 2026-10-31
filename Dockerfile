FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY tests/ tests/

# data/ and models/ are excluded from the image; mount them at runtime:
#   docker run -v ./data:/app/data -v ./models:/app/models <image>
ENV PYTHONPATH=/app

CMD ["python", "-c", "print('Set CMD to your entrypoint, e.g. python -m src.train')"]
