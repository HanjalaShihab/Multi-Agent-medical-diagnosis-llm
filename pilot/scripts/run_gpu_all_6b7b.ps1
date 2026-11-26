<#
.SYNOPSIS
    One command, unattended: run all ten full GPU audits (6B + 7B x 5 datasets).

.DESCRIPTION
    Runs sequentially, one config at a time, never in parallel on the same GPU:
      gpu_6b_pubmedqa -> ... -> gpu_6b_medxpertqa -> gpu_7b_pubmedqa -> ...
    Every run uses its own config and own output directory and always passes
    --resume, so an interrupted run continues instead of starting over.
    Per-run stdout/stderr is teed to logs/gpu_all_6b7b/<name>.log.
    After the last run the shareable summaries are copied to results_summary/.

    Usage (from anywhere, PowerShell):
        powershell -ExecutionPolicy Bypass -File .\scripts\run_gpu_all_6b7b.ps1

    Prerequisites (once, while online):
        python scripts/download_data.py all --out data/raw
        python -m bts_audit.cli prepare --dataset pubmedqa --raw data/raw/pubmedqa --out data/cases/pubmedqa_300.jsonl --n 300
        python -m bts_audit.cli prepare --dataset medqa --raw data/raw/medqa --out data/cases/medqa_300.jsonl --n 300
        python -m bts_audit.cli prepare --dataset medmcqa --raw data/raw/medmcqa --out data/cases/medmcqa_300.jsonl --n 300
        python -m bts_audit.cli prepare --dataset mmlu_med --raw data/raw/mmlu --out data/cases/mmlu_med_300.jsonl --n 300
        python -m bts_audit.cli prepare --dataset medxpertqa --raw data/raw/medxpertqa --out data/cases/medxpertqa_300.jsonl --n 300
        python scripts/prefetch_gpu_models.py 01-ai/Yi-1.5-6B-Chat Qwen/Qwen2.5-7B-Instruct
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"

$models = @("6b", "7b")
$datasets = @("pubmedqa", "medqa", "medmcqa", "mmlu_med", "medxpertqa")

# Preflight: fail early with an actionable message instead of ten dead runs.
foreach ($d in $datasets) {
    $caseFile = "data/cases/${d}_300.jsonl"
    if (-not (Test-Path $caseFile)) {
        throw "Missing $caseFile. Prepare it first (see the header of this script)."
    }
    foreach ($m in $models) {
        $cfg = "configs/gpu_${m}_${d}.yaml"
        if (-not (Test-Path $cfg)) {
            throw "Missing $cfg. Run 'git pull' then 'python scripts/make_gpu_all_configs.py'."
        }
    }
}

$logDir = "logs/gpu_all_6b7b"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$summary = @()
$started = Get-Date
Write-Host "[START] $($started.ToString('o')) - 10 full runs, 300 cases each" -ForegroundColor Cyan

foreach ($m in $models) {
    foreach ($d in $datasets) {
        $name = "gpu_${m}_${d}"
        $cfg = "configs/$name.yaml"
        $log = Join-Path $logDir "$name.log"
        Write-Host "[RUN] $name  $(Get-Date -Format o)  log=$log" -ForegroundColor Cyan

        & python -m bts_audit.cli run --config $cfg --resume 2>&1 | Tee-Object -FilePath $log
        $code = $LASTEXITCODE

        if ($code -ne 0) {
            Write-Host "[FAIL] $name exit=$code - see $log" -ForegroundColor Red
            $summary += "$name FAIL(exit=$code)"
        }
        else {
            Write-Host "[OK]   $name  $(Get-Date -Format o)" -ForegroundColor Green
            $summary += "$name OK"
        }
    }
}

& python scripts/collect_results.py --src results --out results_summary --pattern 'gpu_[67]b_*'

Write-Host "`n[SUMMARY] $(Get-Date -Format o) elapsed=$((Get-Date) - $started)" -ForegroundColor Cyan
$summary | ForEach-Object { Write-Host "  $_" }
Write-Host "Logs: $logDir | Summaries: results_summary/" -ForegroundColor Cyan
