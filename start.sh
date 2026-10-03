#!/usr/bin/env bash
# Starts the 3dGen web app and opens it in your Windows browser.
# The "3dGen" shortcut on your desktop runs this; you never need to call it by hand.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gen3d
cd "$(dirname "$0")"
PORT="${GEN3D_PORT:-7860}"
# Open the browser once the server answers (cmd.exe is Windows, reachable from WSL).
( for _ in $(seq 90); do
    if curl -s "localhost:$PORT" >/dev/null; then
      cmd.exe /c start "http://localhost:$PORT" >/dev/null 2>&1 || true; break
    fi; sleep 1
  done ) &
exec python -m gen3d.web
