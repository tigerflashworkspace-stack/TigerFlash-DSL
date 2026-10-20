@echo off
cd /d "%~dp0"
if not exist "AIWorkspace.exe" (
  echo AIWorkspace.exe nao existe. Execute BUILD_WINDOWS_MINGW.bat primeiro.
  pause
  exit /b 1
)
start "AI Workspace" /D "%~dp0" "%~dp0AIWorkspace.exe" %*
