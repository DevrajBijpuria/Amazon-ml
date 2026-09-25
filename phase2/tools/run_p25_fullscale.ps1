# Phase 2.5 full-target sequence, one job at a time (memory): blocking at 10k S1, model experiments
# on the frozen 10k candidate set, then the B0 resource tests at 25k and 50k S1 (train + validation only).
$tools = $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
& powershell -NoProfile -File "$tools\run_measured.ps1" -Name blocking_10k -PyArgs "phase2/src/phase2_5.py blocking --scale 10k"
& powershell -NoProfile -File "$tools\run_measured.ps1" -Name models_10k -PyArgs "phase2/src/phase2_5_models.py --candset phase2/output/phase2_5/candsets/10k/b0 --scale 10k"
foreach ($n in 25000, 50000) {
    & powershell -NoProfile -File "$tools\run_measured.ps1" -Name "resource_b0_$n" `
        -PyArgs "phase2/src/run_phase2.py --config phase2/output/phase2_5/configs/resource_b0_$n.yaml --mode dev --until model"
}
