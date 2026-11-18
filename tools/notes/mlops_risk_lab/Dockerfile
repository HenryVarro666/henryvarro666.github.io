# Optional teaching example. Not built or runtime-tested for this guide.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLOPS_LAB_ROOT=/app/workspace
COPY requirements.txt requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
COPY lab.py api.py ./
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
