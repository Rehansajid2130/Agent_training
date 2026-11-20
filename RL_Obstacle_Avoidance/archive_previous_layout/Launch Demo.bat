@echo off
cd /d "%~dp0"
python project.py demo
if errorlevel 1 pause
