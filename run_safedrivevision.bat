@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0Caffe\inference.py" --source 0
