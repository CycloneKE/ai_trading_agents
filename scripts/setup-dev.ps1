<#
PowerShell helper to create a fresh dev venv, install minimal test/runtime deps, and run focused tests.

Usage (PowerShell):
  # Remove existing venv (optional)
  Remove-Item -Recurse -Force .\.venv
  # Create new venv and install
  .\scripts\setup-dev.ps1

#>
param()

Write-Host "Setting up development environment..."

python -V
py -3 -m venv .\.venv
Write-Host "Activating virtual env"
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

# Install minimal runtime/test deps used by CI/tests
if (Test-Path requirements-test.txt) {
    pip install -r requirements-test.txt
}

# Recommended extra packages for running backtests and metrics locally
pip install numpy pandas prometheus_client

Write-Host "Running focused tests (live risk manager + backtest integration)"
. .\.venv\Scripts\python.exe -m pytest -q tests/test_live_risk_manager.py tests/test_poc_backtest_integration.py

Write-Host "Dev setup complete. Activate the venv with: . \.venv\Scripts\Activate.ps1" 
