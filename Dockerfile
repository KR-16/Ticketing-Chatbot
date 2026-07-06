FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image a fraction of the CUDA wheel's size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake the NLTK data the preprocessor may need at inference time
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt punkt_tab

COPY src/ src/

RUN useradd --create-home appuser
USER appuser

# The trained bundle is mounted at runtime, keeping the image model-free
ENV BUNDLE_DIR=/app/models/bundle
EXPOSE 8000

# /health answers 503 until a bundle is mounted, so the container reports
# unhealthy (accurately) rather than crash-looping
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
