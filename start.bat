@echo off
rem OFM one-click start (Windows): double-click this file.
rem First run: creates .venv and installs OFM into it. Then opens the studio in your browser.
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.11 이상이 필요합니다 ^(Python 3.11+ required^): https://www.python.org/downloads/
  echo 설치할 때 "Add python.exe to PATH"를 체크하세요.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo 처음 실행: 가상환경^(.venv^)을 만들고 OFM을 설치합니다...
  %PY% -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -c "import ofm, numpy, PIL" >nul 2>nul || (
  ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -e . || goto :fail
)
".venv\Scripts\python.exe" -m ofm studio %*
exit /b 0

:fail
echo 설치에 실패했습니다. 위의 메시지를 확인하세요. ^(Installation failed, see above.^)
pause
exit /b 1
