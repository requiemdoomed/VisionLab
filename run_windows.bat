@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
if exist .venv\Scripts\python.exe goto ready
py -3.12 -c "import sys; sys.exit(0)" >nul 2>&1
if not errorlevel 1 goto create312
py -3.11 -c "import sys; sys.exit(0)" >nul 2>&1
if not errorlevel 1 goto create311
python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
if errorlevel 1 goto no_python
python -m venv .venv
if errorlevel 1 goto fail
goto ready
:create312
py -3.12 -m venv .venv
if errorlevel 1 goto fail
goto ready
:create311
py -3.11 -m venv .venv
if errorlevel 1 goto fail
:ready
.venv\Scripts\python.exe -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 'Use Python 3.11 or 3.12; create a fresh .venv if needed.')"
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto fail
popd
exit /b 0
:no_python
echo Install 64-bit Python 3.11 or 3.12 with PATH enabled, then try again.
:fail
echo Setup or launch failed. Check the error above and README.md.
popd
pause
exit /b 1
