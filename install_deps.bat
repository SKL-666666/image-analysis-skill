@echo off
title image-analysis dependency installer
echo ==========================================
echo   image-analysis dependency installer
echo ==========================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ first:
    echo         https://www.python.org/downloads/
    echo         IMPORTANT: check "Add Python to PATH" during install
    echo.
    echo Press any key to close...
    pause >nul
    exit /b 1
)

set NEED_INSTALL=
set NEED_RAPID=

echo [1/5] Checking pillow ...
python -c "import PIL" >nul 2>nul
if %errorlevel% equ 0 (echo       already installed) else (echo       MISSING & set NEED_INSTALL=1)

echo [2/5] Checking numpy ...
python -c "import numpy" >nul 2>nul
if %errorlevel% equ 0 (echo       already installed) else (echo       MISSING & set NEED_INSTALL=1)

echo [3/5] Checking opencv-python-headless ...
python -c "import cv2" >nul 2>nul
if %errorlevel% equ 0 (echo       already installed) else (echo       MISSING & set NEED_INSTALL=1)

echo [4/5] Checking rapidocr (optional, greatly improves OCR) ...
python -c "import rapidocr_onnxruntime" >nul 2>nul
if %errorlevel% equ 0 (
    echo       already installed
) else (
    echo       optional - not installed
    set /p INSTALL_RAPID="       Install it now? (y/n, default n): "
    if /i "%INSTALL_RAPID%"=="y" set NEED_RAPID=1
)

if "%NEED_INSTALL%"=="" if "%NEED_RAPID%"=="" (
    echo.
    echo ==========================================
    echo   All dependencies already installed!
    echo   Nothing to do. Test with:
    echo   python scripts\analyze_image.py your_image.png --plain
    echo ==========================================
    echo Press any key to close...
    pause >nul
    exit /b 0
)

if not "%NEED_INSTALL%"=="" (
    echo.
    echo [5/5] Installing required packages: pillow numpy opencv-python-headless
    python -m pip install pillow numpy opencv-python-headless
    if %errorlevel% neq 0 (
        echo [ERROR] Install failed. Check your network and retry.
        echo.
        echo Press any key to close...
        pause >nul
        exit /b 1
    )
)

if "%NEED_RAPID%"=="1" (
    echo.
    echo Installing optional package: rapidocr_onnxruntime
    python -m pip install rapidocr_onnxruntime
    if %errorlevel% neq 0 (
        echo [WARN] rapidocr install failed - Windows OCR will still work.
    )
)

echo.
echo ==========================================
echo   Done! Test with:
echo   python scripts\analyze_image.py your_image.png --plain
echo ==========================================
echo Press any key to close...
pause >nul
