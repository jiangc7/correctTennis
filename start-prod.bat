@echo off
rem Production-style local run: builds the frontend once, then FastAPI serves UI + API on http://localhost:8001
rem Use this behind Cloudflare Tunnel (point the tunnel at http://localhost:8001).
set ROOT=%~dp0
cd /d "%ROOT%frontend"
call npm run build
if errorlevel 1 (echo Frontend build failed & pause & exit /b 1)
set PY=python
if exist "C:\Python311\python.exe" set PY=C:\Python311\python.exe
cd /d "%ROOT%backend"
echo.
echo TennisVision AI running on http://localhost:8001  (Ctrl+C to stop)
%PY% -m uvicorn main:app --host 127.0.0.1 --port 8001
pause
