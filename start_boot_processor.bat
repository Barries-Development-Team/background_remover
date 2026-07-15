@echo off
title Boot Photo Processor
echo Waiting for Google Drive to mount...

:waitloop
if not exist "G:\My Drive\Phone uploaded photos" (
    timeout /t 5 /nobreak >nul
    goto waitloop
)

cd /d "C:\Users\Jonathan\OneDrive\Documents\Photo processing tools"
python boot_photo_processor.py
pause