#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ ! -x .venv/bin/python ]; then
  selected_python=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
      selected_python="$candidate"
      break
    fi
  done
  if [ -z "$selected_python" ]; then
    echo "Install Python 3.11 or 3.12, then run this script again." >&2
    exit 1
  fi
  "$selected_python" -m venv .venv
fi
.venv/bin/python -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else "The existing .venv needs Python 3.11 or 3.12. Extract into a fresh folder and try again.")'
.venv/bin/python -m pip install -r requirements.txt
printf '\nOpen http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.\n'
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
