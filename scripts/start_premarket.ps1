# Starts the trading backend + dashboard for a multi-trader testing session.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_premarket.ps1
#   powershell ... start_premarket.ps1 -SkipBuild   # reuse the last frontend build
#
# Backend and frontend run in their own windows; close them (or Ctrl+C in each)
# to stop the session.

param(
    [switch]$SkipBuild,
    [string]$Config = "config\config.json"
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".env")) {
    Write-Error ".env not found. Copy .env.example and set SECRET_KEY plus your data API keys first."
    exit 1
}

# Detect the LAN IPv4 traders will use to reach this machine.
$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.254*' } |
    Sort-Object -Property InterfaceMetric |
    Select-Object -First 1).IPAddress
if (-not $ip) { $ip = 'localhost' }

# The browser's Origin header will be http://<ip>:3001, so the API's CORS
# allowlist must include it or every dashboard request gets blocked.
$env:CORS_ALLOWED_ORIGINS = "http://localhost:3001,http://${ip}:3001"

if (-not $SkipBuild) {
    Write-Host "Building frontend (production)..." -ForegroundColor Cyan
    Push-Location frontend
    npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Error "Frontend build failed."; exit 1 }
    Pop-Location
}

Write-Host "Backfilling NSE history (catches up any missed days)..." -ForegroundColor Cyan
& "$root\.venv\Scripts\python.exe" -m src.connectors.nse_scraper --backfill
if ($LASTEXITCODE -ne 0) { Write-Host "Backfill failed (non-fatal, continuing)" -ForegroundColor Yellow }

Write-Host "Running preflight checks..." -ForegroundColor Cyan
& "$root\.venv\Scripts\python.exe" "$root\scripts\preflight.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Preflight failed - fix the blocking issues above, then rerun."
    exit 1
}

Write-Host "Starting backend (config: $Config)..." -ForegroundColor Cyan
Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "src.agent.main", "--config", $Config `
    -WorkingDirectory $root

Write-Host "Starting dashboard on port 3001..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/k", "npm run start" `
    -WorkingDirectory "$root\frontend"

Write-Host ""
Write-Host "=== Premarket testing session ===" -ForegroundColor Green
Write-Host "Dashboard (traders open this):  http://${ip}:3001"
Write-Host "API:                            http://${ip}:5001/api/health"
Write-Host "Monitoring:                     http://${ip}:8080/health (basic auth)"
Write-Host ""
Write-Host "Add trader logins with:  python scripts\add_traders.py <name1> <name2> ..."
Write-Host "If traders cannot connect, allow ports 3001/5001 through Windows Firewall:"
Write-Host '  netsh advfirewall firewall add rule name="TradingTest" dir=in action=allow protocol=tcp localport=3001,5001'
