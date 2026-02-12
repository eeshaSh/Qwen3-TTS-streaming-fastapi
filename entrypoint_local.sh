#!/bin/bash
set -e

NUM_WORKERS=${TTS_NUM_WORKERS:-2}
BASE_PORT=8001
WORK_DIR="$(pwd)"
LOCAL_RUN_DIR="${WORK_DIR}/.run"
export TTS_VOICE_META_DIR="${TTS_VOICE_META_DIR:-${WORK_DIR}/voices}"

echo "[entrypoint_local] Working directory: ${WORK_DIR}"
echo "[entrypoint_local] Starting with ${NUM_WORKERS} worker(s)..."

# Create local runtime dirs for nginx (avoids writing to /var or /etc)
mkdir -p "${LOCAL_RUN_DIR}/nginx" "${LOCAL_RUN_DIR}/logs"

# Generate nginx upstream servers
UPSTREAM=""
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    UPSTREAM="${UPSTREAM}        server 127.0.0.1:$((BASE_PORT + i));"$'\n'
done

# Write nginx.conf to working directory
cat > "${LOCAL_RUN_DIR}/nginx.conf" << NGINX_EOF
# Run as current user — no 'user' directive, no root needed
worker_processes 1;
pid ${LOCAL_RUN_DIR}/nginx/nginx.pid;
error_log ${LOCAL_RUN_DIR}/logs/nginx_error.log;

events {
    worker_connections 64;
}

http {
    access_log ${LOCAL_RUN_DIR}/logs/nginx_access.log;

    # Temp file paths inside local .run dir
    client_body_temp_path ${LOCAL_RUN_DIR}/nginx/client_body;
    proxy_temp_path ${LOCAL_RUN_DIR}/nginx/proxy;
    fastcgi_temp_path ${LOCAL_RUN_DIR}/nginx/fastcgi;
    uwsgi_temp_path ${LOCAL_RUN_DIR}/nginx/uwsgi;
    scgi_temp_path ${LOCAL_RUN_DIR}/nginx/scgi;

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
directory=${WORK_DIR}
autorestart=true
startretries=999
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
"
done

# Write supervisord.conf to working directory
cat > "${LOCAL_RUN_DIR}/supervisord.conf" << EOF
[supervisord]
nodaemon=true
logfile=${LOCAL_RUN_DIR}/logs/supervisord.log
pidfile=${LOCAL_RUN_DIR}/supervisord.pid

[program:nginx]
command=nginx -e ${LOCAL_RUN_DIR}/logs/nginx_error.log -c ${LOCAL_RUN_DIR}/nginx.conf -g "daemon off;"
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

${WORKERS}
EOF

echo "[entrypoint_local] Generated configs in ${LOCAL_RUN_DIR}/"
echo "[entrypoint_local] Starting supervisord..."

exec supervisord -c "${LOCAL_RUN_DIR}/supervisord.conf"
