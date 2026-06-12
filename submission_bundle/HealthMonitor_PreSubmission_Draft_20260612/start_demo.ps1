$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PYTHONPATH = $root

Write-Host "Starting ECG HealthMonitor dashboard..."
Write-Host "API is optional. To enable live verification, run in another terminal:"
Write-Host "  python -m uvicorn healthmonitor.main:app --host 127.0.0.1 --port 8000"
python -m streamlit run dashboard.py
