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

echo [1/4] Checking pillow ...
python -c "import PIL" >nul 2>nul
if %errorlevel% equ 0 (echo       already installed) else (echo       MISSING & set NEED_INSTALL=1)

echo [2/4] Checking numpy ...
python -c "import numpy" >nul 2>nul
if %errorlevel% equ 0 (echo       already installed) else (echo       MISSING & set NEED_INSTALL=1)

echo [3/4] Checking opencv-python-headless ...
python -c "import cv2" >nul 2>nul
if %errorlevel% equ 0 (echo       already installed) else (echo       MISSING & set NEED_INSTALL=1)

if "%NEED_INSTALL%"=="" (
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

echo.
echo [4/4] Installing missing packages: pillow numpy opencv-python-headless
python -m pip install pillow numpy opencv-python-headless

if %errorlevel% neq 0 (
    echo [ERROR] Install failed. Check your network and retry.
    echo.
    echo Press any key to close...
    pause >nul
    exit /b 1
)

echo.
echo ==========================================
echo   Done! Test with:
echo   python scripts\analyze_image.py your_image.png --plain
echo ==========================================
echo Press any key to close...
pause >nul
