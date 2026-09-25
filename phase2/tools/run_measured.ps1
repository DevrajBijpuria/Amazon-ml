# Runs a python command with an external memory monitor and wall-clock timing.
# Usage: powershell -File run_measured.ps1 -Name <label> -PyArgs "<python args>"
# Writes phase2/output/phase2_5/runs/<label>.log, <label>.mem.csv and <label>.summary.json
param([string]$Name, [string]$PyArgs)
$root = Resolve-Path "$PSScriptRoot\..\.."
$dir = Join-Path $root "phase2\output\phase2_5\runs"
New-Item -ItemType Directory -Force $dir | Out-Null
$log, $err = (Join-Path $dir "$Name.log"), (Join-Path $dir "$Name.err.log")
$env:PYTHONIOENCODING = "utf-8"
$start = Get-Date
$p = Start-Process python -ArgumentList $PyArgs -WorkingDirectory $root -PassThru -NoNewWindow `
    -RedirectStandardOutput $log -RedirectStandardError $err
$null = $p.Handle  # caching the handle makes ExitCode available after exit
$watch = Start-Process powershell -ArgumentList "-NoProfile", "-File", "$PSScriptRoot\memwatch.ps1", `
    "-Out", (Join-Path $dir "$Name.mem.csv"), "-WaitForPid", $p.Id -PassThru -WindowStyle Hidden
$p.WaitForExit()
$watch.WaitForExit()
$mem = Import-Csv (Join-Path $dir "$Name.mem.csv")
$summary = [ordered]@{
    name = $Name; args = $PyArgs; exit_code = $p.ExitCode
    wall_seconds = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    main_process_peak_working_set_mb = ($mem | Measure-Object main_peak_ws_mb -Maximum).Maximum
    all_python_processes_peak_mb = ($mem | Measure-Object all_python_ws_mb -Maximum).Maximum
    samples = $mem.Count
}
$summary | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $dir "$Name.summary.json")
$summary | ConvertTo-Json
