#!/usr/bin/env bash
# AutoMiniLLM - one-click run script (Linux/macOS)
# Starts the FastAPI backend serving the React dashboard on http://localhost:8000
set -e
cd "$(dirname "$0")/.."

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

echo "[2/4] Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "[3/4] Building the React dashboard..."
cd frontend
npm run build
cd ..

echo "[4/4] Seeding demo experiments (real measurements)..."
python scripts/seed_demo.py

echo ""
echo "============================================================"
echo " AutoMiniLLM is ready!"
echo " Starting backend on http://localhost:8000"
echo " (Ctrl+C to stop)"
echo "============================================================"
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
