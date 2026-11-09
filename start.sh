#!/usr/bin/env bash
# OFM one-click start (macOS / Linux): ./start.sh
# First run: creates .venv and installs OFM into it. Then opens the studio in your browser,
# where you make worlds, look at them in 3D and rate them. Extra arguments go to
# `ofm studio` (e.g. ./start.sh --port 9000 --out my_worlds).
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.11 이상이 필요합니다 (Python 3.11+ required): https://www.python.org/downloads/"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "처음 실행: 가상환경(.venv)을 만들고 OFM을 설치합니다…"
  "$PY" -m venv .venv
fi
if ! .venv/bin/python -c 'import ofm, numpy, PIL' >/dev/null 2>&1; then
  .venv/bin/python -m pip install --quiet --disable-pip-version-check -e .
fi
exec .venv/bin/python -m ofm studio "$@"
