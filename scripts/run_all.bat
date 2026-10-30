@echo off
REM AutoMiniLLM - one-click run script (Windows)
REM Starts the FastAPI backend serving the React dashboard on http://localhost:8000
setlocal
cd /d "%~dp0\.."

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt

echo [2/4] Installing frontend dependencies...
cd frontend
call npm install
cd ..

echo [3/4] Building the React dashboard...
cd frontend
call npm run build
cd ..

echo [4/4] Seeding demo experiments (real measurements)...
python scripts\seed_demo.py

echo.
echo ============================================================
echo  AutoMiniLLM is ready!
echo  Starting backend on http://localhost:8000
echo  (Ctrl+C to stop)
echo ============================================================
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
