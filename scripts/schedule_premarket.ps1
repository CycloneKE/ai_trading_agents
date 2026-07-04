# Registers (or removes) a Windows scheduled task that starts the trading
# stack automatically on weekday mornings, so the session is already running
# when traders arrive for premarket.
#
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_premarket.ps1                    # 03:45 daily Mon-Fri
#   powershell ... schedule_premarket.ps1 -At 04:00
#   powershell ... schedule_premarket.ps1 -Remove
#   powershell ... schedule_premarket.ps1 -Status
#
# Times are LOCAL machine time - premarket is 4:00 AM US Eastern, so convert
# for your timezone (e.g. 11:00 AM in Nairobi during EDT).
#
# Run from an elevated (Administrator) PowerShell.

param(
    [string]$At = "03:45",
    [switch]$Remove,
    [switch]$Status
)

$taskName = "AITrading-PremarketSession"
$root = Split-Path -Parent $PSScriptRoot
$startScript = Join-Path $PSScriptRoot "start_premarket.ps1"

if ($Status) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($task) {
        $info = $task | Get-ScheduledTaskInfo
        Write-Host "Task '$taskName': $($task.State)"
        Write-Host "  Next run:  $($info.NextRunTime)"
        Write-Host "  Last run:  $($info.LastRunTime) (result $($info.LastTaskResult))"
    } else {
        Write-Host "Task '$taskName' is not registered."
    }
    exit 0
}

if ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task '$taskName' (if it existed)."
    exit 0
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$startScript`" -SkipBuild" `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $At

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)   # 0 = no time limit; session runs until stopped

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Start AI trading stack for premarket testing" -Force | Out-Null

Write-Host "Registered '$taskName' to run Mon-Fri at $At (local time)."
Write-Host "Uses -SkipBuild: run 'npm run build' in frontend/ after frontend changes."
Write-Host "Check with:  schedule_premarket.ps1 -Status    Remove with:  -Remove"
