# Samples the memory of every python process every 2 s until the watched process exits.
# Usage: powershell -File memwatch.ps1 -Out <csv> -WaitForPid <pid>
# Writes: timestamp, main-process working set (MB), all python processes total (MB).
param([string]$Out, [int]$WaitForPid)
"time,main_ws_mb,main_peak_ws_mb,all_python_ws_mb" | Out-File -Encoding utf8 $Out
while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
    $main = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    $all = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum
    if ($main) {
        "{0},{1},{2},{3}" -f (Get-Date -Format s), [int]($main.WorkingSet64 / 1MB), [int]($main.PeakWorkingSet64 / 1MB), [int]($all / 1MB) |
            Out-File -Append -Encoding utf8 $Out
    }
    Start-Sleep -Seconds 2
}
