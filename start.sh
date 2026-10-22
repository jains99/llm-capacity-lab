#!/usr/bin/env bash
# Start the Capacity Lab dev server, wait until it is ready, then open it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-5173}"
URL="http://localhost:${PORT}/"
PID_FILE="$ROOT/.capacity-lab.pid"
LOG_FILE="$ROOT/.capacity-lab.log"
VITE_BIN="$ROOT/node_modules/.bin/vite"

open_url() {
  if [[ "${NO_OPEN:-0}" == "1" ]]; then
    printf 'Open %s\n' "$URL"
    return 0
  fi

  case "$(uname -s)" in
    Darwin)
      if command -v open >/dev/null 2>&1; then
        open "$URL"
        return 0
      fi
      ;;
    Linux)
      if grep -qi microsoft /proc/version 2>/dev/null; then
        if command -v cmd.exe >/dev/null 2>&1 && cmd.exe /c start "$URL" >/dev/null 2>&1; then
          return 0
        fi
      fi
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL"
        return 0
      fi
      ;;
  esac

  printf 'Open %s\n' "$URL"
}

is_our_server() {
  curl -fsS --max-time 2 "$URL" 2>/dev/null | grep -q "Capacity Lab"
}

if is_our_server; then
  printf 'Capacity Lab is already running at %s\n' "$URL"
  open_url
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    printf 'A Capacity Lab process is running (PID %s) but is not responding.\n' "$OLD_PID"
    printf 'Run ./stop.sh first, then try again.\n'
    exit 1
  fi
  rm -f "$PID_FILE"
fi

if [[ ! -x "$VITE_BIN" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    printf 'npm is not installed. Install Node.js 20+ first.\n' >&2
    exit 1
  fi
  printf 'Dependencies are missing. Installing with npm...\n'
  npm install
fi

printf 'Starting Capacity Lab on port %s...\n' "$PORT"
nohup "$VITE_BIN" --port "$PORT" --strictPort >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$PID_FILE"

READY=0
for _ in $(seq 1 80); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  if is_our_server; then
    READY=1
    break
  fi
  sleep 0.25
done

if [[ "$READY" != "1" ]]; then
  printf 'Capacity Lab did not start.\n' >&2
  if [[ -f "$LOG_FILE" ]]; then
    printf '\nLast server output:\n' >&2
    tail -n 20 "$LOG_FILE" >&2 || true
  fi
  rm -f "$PID_FILE"
  exit 1
fi

printf 'Capacity Lab is running at %s (PID %s)\n' "$URL" "$SERVER_PID"
printf 'Log: %s\n' "$LOG_FILE"
open_url
