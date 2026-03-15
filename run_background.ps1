# Vehicle Counter - Run in Background
# Chạy service nền 24/7

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vehicle Counter Background Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Python
$pythonPath = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonPath) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Starting service in background..." -ForegroundColor Green
Write-Host "Log file: vehicle_counter.log" -ForegroundColor Yellow
Write-Host ""

# Chạy trong background với nohup-like behavior
$process = Start-Process -FilePath "python" `
    -ArgumentList "vehicle_counter_service.py" `
    -WorkingDirectory $scriptPath `
    -RedirectStandardOutput "service_output.log" `
    -RedirectStandardError "service_error.log" `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Service started with PID: $($process.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "To stop the service:" -ForegroundColor Yellow
Write-Host "  Stop-Process -Id $($process.Id)" -ForegroundColor White
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Yellow
Write-Host "  Get-Content vehicle_counter.log -Tail 50 -Wait" -ForegroundColor White
Write-Host ""

# Lưu PID để có thể stop sau
$process.Id | Out-File "service.pid"
Write-Host "PID saved to service.pid" -ForegroundColor Gray
