@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo AI Workspace - Windows build (mantem a configuracao original)
echo ============================================================

where g++ >nul 2>nul
if errorlevel 1 (
  echo ERRO: g++ nao encontrado.
  echo Instale MinGW-w64 com modelo de threads POSIX e adicione bin ao PATH.
  echo Consulte README_WINDOWS_PT-BR.md.
  pause
  exit /b 1
)

if not defined RAYLIB_DIR set "RAYLIB_DIR=%~dp0raylib"
if not exist "%RAYLIB_DIR%\src\raylib.h" (
  echo ERRO: nao encontrei "%RAYLIB_DIR%\src\raylib.h"
  echo Coloque o pacote de desenvolvimento raylib em uma pasta chamada raylib
  echo ao lado deste script, ou defina RAYLIB_DIR para a pasta da raylib.
  echo Exemplo: set RAYLIB_DIR=C:\raylib\raylib
  pause
  exit /b 1
)

set "RAYLIB_LIB_DIR="
if exist "%RAYLIB_DIR%\src\libraylib.a" set "RAYLIB_LIB_DIR=%RAYLIB_DIR%\src"
if not defined RAYLIB_LIB_DIR if exist "%RAYLIB_DIR%\lib\libraylib.a" set "RAYLIB_LIB_DIR=%RAYLIB_DIR%\lib"
if not defined RAYLIB_LIB_DIR if exist "%RAYLIB_DIR%\lib\lib/libraylib.a" set "RAYLIB_LIB_DIR=%RAYLIB_DIR%\lib\lib"
if not defined RAYLIB_LIB_DIR (
  echo ERRO: nao encontrei libraylib.a em src ou lib dentro de RAYLIB_DIR.
  echo Este script requer raylib compilada para o mesmo MinGW-w64 instalado.
  echo Nao use uma raylib.lib feita para MSVC com este compilador.
  pause
  exit /b 1
)

echo Compilando com MULTITHREAD=1, 1280x720, 60 FPS e MSAA 4x do codigo original...
g++ "AiWorskspace.cpp" -o "AIWorkspace.exe" -O2 -std=c++17 -pthread -DMULTITHREAD=1 -I"%RAYLIB_DIR%\src" -L"%RAYLIB_LIB_DIR%" -lraylib -lopengl32 -lgdi32 -lwinmm
if errorlevel 1 (
  echo.
  echo FALHA na compilacao. Leia README_WINDOWS_PT-BR.md e confira se raylib e MinGW
  echo possuem a mesma arquitetura (32-bit ou 64-bit) e se MinGW usa POSIX threads.
  pause
  exit /b 1
)

echo.
echo Build concluido: AIWorkspace.exe
 echo Para executar, use RUN_WINDOWS.bat
pause
exit /b 0
