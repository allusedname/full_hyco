@echo off
REM  get_coco_test.bat  –  COCO-2017 *test* images only (Windows)

setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"

:: ----------------------------------------------------------
:: CONFIG
:: ----------------------------------------------------------
set "ZIP=test2017.zip"
set "MINSIZE=6300000000"           REM 6.3 GB – sanity check
set "IMAGES_URL=http://images.cocodataset.org/zips/%ZIP%"
set "IMG_DIR=coco\images"

:: ----------------------------------------------------------
:: 1. Create destination folder
:: ----------------------------------------------------------
if not exist "%IMG_DIR%" mkdir "%IMG_DIR%"

:: ----------------------------------------------------------
:: 2. Download with resume support (curl -C -)
:: ----------------------------------------------------------
echo(
echo ===== Downloading %ZIP% (resume supported) =====
curl -L -C - "%IMAGES_URL%" -o "%ZIP%"
if errorlevel 1 (
    echo *** curl reported an error – aborting. ***
    exit /b 1
)

:: ----------------------------------------------------------
:: 3. Verify size
:: ----------------------------------------------------------
for %%S in ("%ZIP%") do set "ZSIZE=%%~zS"
if !ZSIZE! LSS !MINSIZE! (
    echo *** ERROR: %ZIP% is incomplete (!ZSIZE! bytes < !MINSIZE!). ***
    echo *** Re-run this script to resume the download.               ***
    exit /b 1
)

:: ----------------------------------------------------------
:: 4. Extract and clean up
:: ----------------------------------------------------------
echo ----- Extracting %ZIP% -----
powershell -NoLogo -Command ^
  "Expand-Archive -Path '%ZIP%' -DestinationPath '%IMG_DIR%' -Force"
del "%ZIP%"

echo(
echo ============ test2017 ready at "%IMG_DIR%\test2017" ============
pause
endlocal
