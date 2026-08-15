\
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "[1/2] Environment check"
python3 00_check_environment.py

echo
echo "[2/2] Model smoke test"
python3 02_smoke_test.py
