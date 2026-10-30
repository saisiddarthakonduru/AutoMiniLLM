@echo off
rem =====================================================
rem  AutoMiniLLM - Stop backend (8000) and frontend (5173)
rem =====================================================

echo Stopping backend on port 8000 ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo Stopping frontend on port 5173 ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo Done. Both servers stopped.