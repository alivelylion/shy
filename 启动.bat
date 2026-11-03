@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PATH=%CD%\.venv;%CD%\.venv\Library\bin;%PATH%"
set "SEE_THROUGH_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "SEE_THROUGH_PYTHON=.venv\Scripts\python.exe"
if exist ".venv\python.exe" set "SEE_THROUGH_PYTHON=.venv\python.exe"
"%SEE_THROUGH_PYTHON%" -u main.py %*
set "task_exit_code=%errorlevel%"
if "%~1"=="" pause
exit /b %task_exit_code%
