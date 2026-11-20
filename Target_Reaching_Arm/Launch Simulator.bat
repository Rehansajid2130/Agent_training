@echo off
cd /d "%~dp0"
python arm_simulator.py
if errorlevel 1 pause
