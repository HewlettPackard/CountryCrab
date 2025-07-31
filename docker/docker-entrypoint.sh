#!/usr/bin/env bash
set -e

# If the user runs: `docker run … image shell`
if [ "$1" = "shell" ]; then
  echo "🔹 Starting container in SHELL mode…"
  exec bash

# Otherwise if they run with "lab" or no args, start Jupyter Lab
elif [ "$1" = "lab" ] || [ -z "$1" ]; then
  echo "🔹 Launching Jupyter Lab on 0.0.0.0:8888"
  echo "👉 Token-based auth as per your jupyter_lab_config.py"
  exec jupyter lab \
    --config=/etc/jupyter/jupyter_lab_config.py \
    --ip=0.0.0.0 \
    --allow-root

# If they passed any other command (e.g. `docker run … image python script.py`), just exec it
else
  exec "$@"
fi