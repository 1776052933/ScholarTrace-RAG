@echo off
cd /d "%~dp0.."
set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=D:\Application files\anaconda\envs\pywork\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
echo M2 requires the FastAPI server first (default: http://127.0.0.1:8000).
"%PYTHON_EXE%" -m streamlit run "Auto_Paper_UI\ui_app.py"
