# Vehicle Counter Service Control
# Điều khiển service dễ dàng

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status", "restart", "log")]
    [string]$Command = "status"
)

$taskName = "VehicleCounterService"
$logFile = "D:\LuanVan\web\streamlit\vehicle_counter.log"

function Show-Status {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        $info = Get-ScheduledTaskInfo -TaskName $taskName
        Write-Host ""
        Write-Host "📊 Vehicle Counter Service Status" -ForegroundColor Cyan
        Write-Host "=================================" -ForegroundColor Cyan
        Write-Host "Task Name    : $taskName" -ForegroundColor White
        Write-Host "State        : $($task.State)" -ForegroundColor $(if($task.State -eq "Running"){"Green"}else{"Yellow"})
        Write-Host "Last Run     : $($info.LastRunTime)" -ForegroundColor White
        Write-Host "Last Result  : $($info.LastTaskResult)" -ForegroundColor White
        Write-Host "Next Run     : $($info.NextRunTime)" -ForegroundColor White
        Write-Host ""
        
        # Kiểm tra process python
        $pythonProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*vehicle_counter*"}
        if ($pythonProcs) {
            Write-Host "🟢 Python Process: Running (PID: $($pythonProcs.Id -join ', '))" -ForegroundColor Green
        } else {
            Write-Host "🔴 Python Process: Not found" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ Task '$taskName' không tồn tại" -ForegroundColor Red
        Write-Host "Chạy setup_task_scheduler.ps1 để tạo task" -ForegroundColor Yellow
    }
}

function Start-Service {
    Write-Host "🚀 Starting service..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Show-Status
}

function Stop-Service {
    Write-Host "⏹️ Stopping service..." -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    
    # Kill python process nếu còn
    $pythonProcs = Get-Process python -ErrorAction SilentlyContinue
    if ($pythonProcs) {
        $pythonProcs | Stop-Process -Force
        Write-Host "Python processes killed" -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 1
    Show-Status
}

function Show-Log {
    if (Test-Path $logFile) {
        Write-Host "📋 Last 50 lines of log:" -ForegroundColor Cyan
        Write-Host "========================" -ForegroundColor Cyan
        Get-Content $logFile -Tail 50
        Write-Host ""
        Write-Host "Để xem log realtime: Get-Content '$logFile' -Tail 50 -Wait" -ForegroundColor Gray
    } else {
        Write-Host "❌ Log file không tồn tại: $logFile" -ForegroundColor Red
    }
}

# Main
switch ($Command) {
    "start" { Start-Service }
    "stop" { Stop-Service }
    "status" { Show-Status }
    "restart" { 
        Stop-Service
        Start-Sleep -Seconds 2
        Start-Service 
    }
    "log" { Show-Log }
}
