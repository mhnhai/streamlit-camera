# Setup Windows Task Scheduler cho Vehicle Counter Service
# Chạy script này với quyền Administrator

$taskName = "VehicleCounterService"
$scriptPath = "D:\LuanVan\web\streamlit"
$pythonPath = "D:\LuanVan\web\streamlit\.venv\Scripts\python.exe"
$scriptFile = "vehicle_counter_service.py"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Vehicle Counter Task Scheduler" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Python path
if (-not (Test-Path $pythonPath)) {
    Write-Host "Python không tìm thấy tại: $pythonPath" -ForegroundColor Red
    Write-Host "Hãy sửa đường dẫn pythonPath trong script này" -ForegroundColor Yellow
    
    # Thử tìm python trong PATH
    $pythonInPath = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonInPath) {
        Write-Host "Tìm thấy Python tại: $($pythonInPath.Source)" -ForegroundColor Green
        $pythonPath = $pythonInPath.Source
    } else {
        exit 1
    }
}

Write-Host "Python: $pythonPath" -ForegroundColor Green
Write-Host "Script: $scriptPath\$scriptFile" -ForegroundColor Green
Write-Host ""

# Xóa task cũ nếu có
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Xóa task cũ..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Tạo action
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $scriptFile -WorkingDirectory $scriptPath

# Tạo trigger - chạy khi khởi động máy
$trigger = New-ScheduledTaskTrigger -AtStartup

# Tạo settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

# Tạo principal (chạy với user hiện tại)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest

# Đăng ký task
try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Vehicle Counter Service - Chạy 24/7 đếm xe từ camera IP"
    
    Write-Host ""
    Write-Host "✅ Task đã được tạo thành công!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Name: $taskName" -ForegroundColor White
    Write-Host "Trigger: Khi khởi động Windows" -ForegroundColor White
    Write-Host ""
    Write-Host "Các lệnh hữu ích:" -ForegroundColor Cyan
    Write-Host "  Start-ScheduledTask -TaskName '$taskName'    # Chạy ngay" -ForegroundColor White
    Write-Host "  Stop-ScheduledTask -TaskName '$taskName'     # Dừng" -ForegroundColor White
    Write-Host "  Get-ScheduledTask -TaskName '$taskName'      # Xem trạng thái" -ForegroundColor White
    Write-Host ""
    
    # Hỏi có muốn chạy ngay không
    $runNow = Read-Host "Bạn có muốn chạy task ngay bây giờ không? (y/n)"
    if ($runNow -eq "y") {
        Start-ScheduledTask -TaskName $taskName
        Write-Host "✅ Task đã được khởi chạy!" -ForegroundColor Green
    }
    
} catch {
    Write-Host "❌ Lỗi khi tạo task: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Mở Task Scheduler để xem: taskschd.msc" -ForegroundColor Gray
