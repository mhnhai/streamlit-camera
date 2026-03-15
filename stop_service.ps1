# Stop Vehicle Counter Service

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

if (Test-Path "service.pid") {
    $pid = Get-Content "service.pid"
    Write-Host "Stopping service with PID: $pid" -ForegroundColor Yellow
    
    try {
        Stop-Process -Id $pid -Force
        Write-Host "Service stopped successfully" -ForegroundColor Green
        Remove-Item "service.pid"
    }
    catch {
        Write-Host "Service may already be stopped" -ForegroundColor Red
        Remove-Item "service.pid" -ErrorAction SilentlyContinue
    }
}
else {
    Write-Host "No service.pid found. Service may not be running." -ForegroundColor Yellow
    
    # Thử tìm process đang chạy
    $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*vehicle_counter_service*" }
    
    if ($processes) {
        Write-Host "Found running processes:" -ForegroundColor Yellow
        $processes | ForEach-Object {
            Write-Host "  PID: $($_.Id)" -ForegroundColor White
        }
        
        $confirm = Read-Host "Stop all? (y/n)"
        if ($confirm -eq "y") {
            $processes | Stop-Process -Force
            Write-Host "All processes stopped" -ForegroundColor Green
        }
    }
}
