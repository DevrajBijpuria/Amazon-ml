# After the B0 resource tests finish: cap/cutoff sweep on the exact frozen rule set at 10k S1 / full targets.
$tools = $PSScriptRoot
$runs = Join-Path $tools "..\output\phase2_5\runs"
while (-not (Test-Path (Join-Path $runs "resource_b0_50000.summary.json"))) { Start-Sleep -Seconds 20 }
& powershell -NoProfile -File "$tools\run_measured.ps1" -Name capsweep_frozen_10k -PyArgs "phase2/src/phase2_5.py capsweep --scale 10k"
