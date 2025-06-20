@echo off
REM ============================================================
REM  get_coco.bat – COCO-2017 dataset downloader (Windows)
REM  Equivalent to Ultralytics YOLOv5 scripts/get_coco.sh
REM ============================================================

REM 1. Make sure we run from the folder that contains this script
cd /d %~dp0

REM ---------- LABEL FILES ------------------------------------
set "LABELS_ZIP=coco2017labels.zip"
set "LABELS_URL=https://github.com/ultralytics/yolov5/releases/download/v1.0/%LABELS_ZIP%"

@REM echo(
@REM echo ===== Downloading COCO-2017 label archive =====
@REM curl -L "%LABELS_URL%" -o "%LABELS_ZIP%"
@REM echo ----- Extracting %LABELS_ZIP% -----
@REM powershell -NoLogo -Command ^
@REM     "Expand-Archive -Path '%LABELS_ZIP%' -DestinationPath '.' -Force"
@REM del "%LABELS_ZIP%"

REM ---------- IMAGE FILES ------------------------------------
set "IMG_DIR=coco\images"
if not exist "%IMG_DIR%" mkdir "%IMG_DIR%"

set "IMAGES_BASE=http://images.cocodataset.org/zips"
for %%Z in (val2017.zip test2017.zip) do (
    echo(
    echo ===== Downloading %%Z =====
    curl -L "%IMAGES_BASE%/%%Z" -o "%%Z"
    echo ----- Extracting %%Z -----
    powershell -NoLogo -Command ^
        "Expand-Archive -Path '%%Z' -DestinationPath '%IMG_DIR%' -Force"
    del "%%Z"
)

echo(
echo ============ All downloads finished! =============
pause
