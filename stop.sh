#!/usr/bin/env bash
# Stop the Capacity Lab dev server started by start.sh.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-5173}"
PID_FILE="$ROOT/.capacity-lab.pid"
STOPPED=0

stop_pid() {
  local pid="$1"
  local label="${2:-Capacity Lab}"

  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  printf 'Stopping %s (PID %s)...\n' "$label" "$pid"
  kill "$pid" 2>/dev/null || true

  for _ in $(seq 1 25); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done

  if kill -0 "$pid" 2>/dev/null; then
    printf 'Process %s did not exit; forcing it.\n' "$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  STOPPED=1
}

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  stop_pid "$PID" "Capacity Lab"
  rm -f "$PID_FILE"
fi

# Fallback for a server that was started manually: only kill a Vite process
# that belongs to this project. Never kill an unrelated service on the port.
if command -v lsof >/dev/null 2>&1; then
  PORT_PIDS="$(lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  for PID in $PORT_PIDS; do
    CMD="$(ps -p "$PID" -o command= 2>/dev/null || true)"
    if [[ "$CMD" == *"$ROOT"* ]]; then
      stop_pid "$PID" "Capacity Lab"
    fi
  done
fi

if [[ "$STOPPED" == "1" ]]; then
  printf 'Capacity Lab stopped.\n'
else
  printf 'No Capacity Lab server was running.\n'
fi
