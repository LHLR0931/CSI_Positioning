@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "LOG=backend\artifacts\replay\export_log.txt"
where conda >nul 2>&1
if errorlevel 1 (
  echo NOT FOUND: conda > "%LOG%"
  echo 找不到 Conda，请先安装并创建 csi 环境。
  pause
  exit /b 1
)
echo ENVIRONMENT=csi > "%LOG%"
conda run -n csi python -X utf8 -m backend.training.export_replay --seed 42 >> "%LOG%" 2>&1
echo EXIT=%ERRORLEVEL% >> "%LOG%"
type "%LOG%"
echo.
echo 完成，可以关闭这个窗口。
pause
