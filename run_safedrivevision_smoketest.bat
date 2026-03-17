@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0Caffe\inference.py" --source blank --no-display --max-frames 1 --save-output "%~dp0smoke-test-output.jpg"
