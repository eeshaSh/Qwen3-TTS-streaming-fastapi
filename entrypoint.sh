#!/bin/bash
set -e

NUM_WORKERS=${TTS_NUM_WORKERS:-2}
BASE_PORT=8001

echo "[entrypoint] Starting with ${NUM_WORKERS} worker(s)..."

# Generate nginx upstream servers
UPSTREAM=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    UPSTREAM="${UPSTREAM}        server 127.0.0.1:$((BASE_PORT + i));"$'\n'
done

# Write nginx.conf directly (no sed/template dependency)
cat > /etc/nginx/nginx.conf << NGINX_EOF
events {
    worker_connections 64;
}

http {
    upstream tts_workers {
        least_conn;
${UPSTREAM}    }

    server {
        listen 8000;

        location / {
            proxy_pass http://tts_workers;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            chunked_transfer_encoding on;
            proxy_connect_timeout 10s;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
        }
    }
}
NGINX_EOF

# Generate supervisord worker sections
WORKERS=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    WORKERS="${WORKERS}
[program:tts-worker-${i}]
command=uvicorn fastapi_tts_server:app --host 127.0.0.1 --port ${PORT}
directory=/app
autorestart=true
startretries=999
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
"
done

# Write final supervisord.conf
cat > /etc/supervisord.conf << EOF
[supervisord]
nodaemon=true
logfile=/dev/stdout
logfile_maxbytes=0

[program:nginx]
command=nginx -g "daemon off;"
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

${WORKERS}
EOF

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

nvidia-cuda-mps-control -d

echo "[entrypoint] Generated nginx config with ${NUM_WORKERS} upstream(s)"
echo "[entrypoint] Starting supervisord..."

exec supervisord -c /etc/supervisord.conf
