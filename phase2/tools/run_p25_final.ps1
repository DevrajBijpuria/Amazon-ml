# Phase 2.5 final sequence, started after run_p25_fullscale.ps1 finishes (one job at a time for memory):
# a dev-scale end-to-end smoke test of the frozen configuration, then the 50k-S1 / full-target runs of
# the blocking-only variant (B0 model) and of the frozen configuration, on the same 50k sample as B0.
$tools = $PSScriptRoot
$runs = Join-Path $tools "..\output\phase2_5\runs"
while (-not (Test-Path (Join-Path $runs "resource_b0_50000.summary.json"))) { Start-Sleep -Seconds 20 }
& powershell -NoProfile -File "$tools\run_measured.ps1" -Name frozen_dev_smoke `
    -PyArgs "phase2/src/run_phase2.py --config phase2/config/phase2_5_frozen.yaml --mode dev"
foreach ($v in "blocking_lr_50000", "frozen_50000") {
    & powershell -NoProfile -File "$tools\run_measured.ps1" -Name $v `
        -PyArgs "phase2/src/run_phase2.py --config phase2/output/phase2_5/configs/$v.yaml --mode dev --until model"
}
