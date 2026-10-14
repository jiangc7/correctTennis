@echo off
rem Starts TennisVision backend (FastAPI :8001) and frontend (Vite :5173) in two windows
set ROOT=%~dp0
set PY=python
if exist "C:\Python311\python.exe" set PY=C:\Python311\python.exe
start "TennisVision backend :8001" cmd /k "cd /d %ROOT%backend && %PY% -m uvicorn main:app --host 0.0.0.0 --port 8001"
start "TennisVision frontend :5173" cmd /k "cd /d %ROOT%frontend && npm run dev"
