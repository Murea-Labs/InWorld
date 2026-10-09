# Murea Agent: one image with the API server and the prebuilt front end (frontend/dist).
#
#   docker build -t murea-agent .
#   docker run -p 127.0.0.1:8080:8080 --add-host host.docker.internal:host-gateway \
#     --env-file .env -v $PWD/data:/app/data murea-agent
#
# A Murea-Fast service on the host is MUREA_URL=http://host.docker.internal:<port>.

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "imageio-ffmpeg>=0.5,<1"
COPY murea_agent/ murea_agent/
COPY app/ app/
COPY scripts/ scripts/
COPY examples/ examples/
COPY config.example.yaml ./
COPY frontend/dist frontend/dist
RUN mkdir -p data
EXPOSE 8080
# Listen on all interfaces inside the container regardless of config.yaml.
CMD ["python", "-c", "import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=8080)"]
