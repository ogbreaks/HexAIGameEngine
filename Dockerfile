FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y curl apt-transport-https gnupg && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

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
