@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where g++ >nul 2>nul
if errorlevel 1 (
  echo ERRO: g++ nao encontrado. Instale MinGW-w64 e adicione-o ao PATH.
  pause
  exit /b 1
)
if not defined RAYLIB_DIR set "RAYLIB_DIR=%~dp0raylib"
if not exist "%RAYLIB_DIR%\src\raylib.h" (
  echo ERRO: nao encontrei "%RAYLIB_DIR%\src\raylib.h". Consulte README_WINDOWS_PT-BR.md.
  pause
  exit /b 1
)
set "RAYLIB_LIB_DIR="
if exist "%RAYLIB_DIR%\src\libraylib.a" set "RAYLIB_LIB_DIR=%RAYLIB_DIR%\src"
if not defined RAYLIB_LIB_DIR if exist "%RAYLIB_DIR%\lib\libraylib.a" set "RAYLIB_LIB_DIR=%RAYLIB_DIR%\lib"
if not defined RAYLIB_LIB_DIR (
  echo ERRO: nao encontrei libraylib.a em src ou lib. A raylib deve ser compilada para MinGW.
  pause
  exit /b 1
)

echo Compilando a versao de uma thread: todos os recursos e qualidade grafica permanecem; so o enxame roda em uma thread.
g++ "AiWorskspace.cpp" -o "AIWorkspace.exe" -O2 -std=c++17 -DMULTITHREAD=0 -I"%RAYLIB_DIR%\src" -L"%RAYLIB_LIB_DIR%" -lraylib -lopengl32 -lgdi32 -lwinmm
if errorlevel 1 (
  echo FALHA. Confira o compilador e a raylib; veja README_WINDOWS_PT-BR.md.
  pause
  exit /b 1
)
echo Build concluido: AIWorkspace.exe
pause
