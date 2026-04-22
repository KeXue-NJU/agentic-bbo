# Build from ``bbo/tasks/surrogate`` (parent of this directory):
#   docker build -f docker/Dockerfile -t agentbbo-surrogate-http-py37:v1 .
# Includes ``assets/`` (joblib + knobs json for decode paths on the host; server reads joblib from here).
FROM python:3.7-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8090 \
    SURROGATE_ASSETS_DIR=/app/assets

WORKDIR /app

COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY docker/server.py /app/server.py
COPY knob_space.py /app/knob_space.py
COPY assets /app/assets

EXPOSE 8090

CMD ["python", "/app/server.py"]
