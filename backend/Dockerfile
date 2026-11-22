FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY prompts/ ./prompts/

ENV HOST=0.0.0.0 \
    PORT=8080 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# 单 worker：每次请求独立，已验证 pair 走外部存储，横向扩展靠多副本
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
