@echo off
REM AutoMiniLLM - build frontend, seed demo, and start server in background
cd /d "%~dp0\.."

echo [1/4] Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo ERROR: npm install failed
    exit /b 1
)

echo [2/4] Building React dashboard...
call npm run build
if errorlevel 1 (
    echo ERROR: npm run build failed
    exit /b 1
)
cd ..

echo [3/4] Seeding demo experiments (real measurements)...
python scripts\seed_demo.py
if errorlevel 1 (
    echo WARNING: seed_demo failed, continuing anyway
)

echo [4/4] Starting AutoMiniLLM server on http://localhost:8000
start "AutoMiniLLM" cmd /c "python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"
echo Server starting in background. Open http://localhost:8000 in your browser.
timeout /t 3 >nul
