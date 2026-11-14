FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9696

WORKDIR /app
RUN pip install --no-cache-dir \
    "fastapi" \
    "uvicorn[standard]" \
    "scikit-learn==1.9.1"

COPY model.py predict.py burn_out_model.bin ./
COPY static ./static
EXPOSE 9696
CMD ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port ${PORT}"]