@echo off
rem =====================================================
rem  AutoMiniLLM - One-click start (backend + frontend)
rem  Double-click this file, or run it from a terminal.
rem =====================================================
cd /d "%~dp0"

echo.
echo [1/4] Starting FastAPI backend on http://localhost:8000 ...
start "AutoMiniLLM Backend" cmd /k "python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"

rem Give the backend a moment to boot
timeout /t 4 /nobreak >nul

echo [2/4] Starting React dashboard (Vite) on http://localhost:5173 ...
cd frontend
start "AutoMiniLLM Frontend" cmd /k "npm run dev"
cd /d "%~dp0"

echo [3/4] Waiting for the dashboard to come up ...
timeout /t 6 /nobreak >nul

echo [4/4] Opening the dashboard in your browser ...
start http://localhost:5173

echo.
echo Done! Backend is on port 8000, dashboard on port 5173.
echo Close the two new terminal windows (or run stop.bat) to shut the servers down.
echo.