FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-training.txt .
RUN iconv -f UTF-16LE -t UTF-8 requirements-training.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

RUN mkdir -p training/models training/checkpoints training/results

COPY game/ game/
COPY server/ server/
COPY training/ training/
COPY config/ config/
COPY main.py .
COPY metrics_server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8080

HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

CMD ["./entrypoint.sh"]
