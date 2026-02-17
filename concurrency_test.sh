#!/bin/bash
HOST="${1:-http://localhost:8000}"
N="${2:-2}"
URL="${HOST}/v1/audio/speech"
BODY='{"input": "Hey I am Eesha. I was born on August 22nd and this is my story. I graduated college and worked at Applied for a while, then I switched jobs to Tesla and now I am a voice in Leons laptop."}'

echo "Launching $N concurrent requests..."
for i in $(seq 1 "$N"); do
    curl -N -s -w "\n[REQ${i}] TTFB: %{time_starttransfer}s Total: %{time_total}s\n" -X POST "$URL" -H "Content-Type: application/json" -d "$BODY" -o /dev/null &
done
wait
echo "Done."
