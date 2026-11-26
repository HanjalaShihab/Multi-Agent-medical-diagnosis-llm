# Fast 6B + 7B PubMedQA probes - one model per GPU, in parallel.
# Run from the pilot/ directory with the venv active.
#
#   .\scripts\run_gpu_quick_6b7b.ps1
#
# 40 cases x H1,H4,H5,SYC x 96 new tokens, 4-bit, HF offline after prefetch.
# Separate configs + output dirs; nothing mixes with the 300-case full runs.
param(
    [string]$Device6B = "0",
    [string]$Device7B = "1",
    [switch]$SkipPrefetch
)

$ErrorActionPreference = "Stop"
$wd = (Get-Location).Path

if (-not $SkipPrefetch) {
    Write-Host "Prefetching 6B + 7B files once while online (avoids 429 waits later)..." -ForegroundColor Yellow
    python scripts\prefetch_gpu_models.py "01-ai/Yi-1.5-6B-Chat" "Qwen/Qwen2.5-7B-Instruct"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Prefetch failed. Check internet access, then rerun." -ForegroundColor Red
        exit 1
    }
}

$job6 = Start-Job -Name "quick-6b" -ScriptBlock {
    param($dev, $wd)
    Set-Location $wd
    $env:CUDA_VISIBLE_DEVICES = $dev
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    python -m bts_audit.cli run `
        --config "configs/gpu_quick_pubmedqa_6b.yaml" `
        --data "data/cases/pubmedqa_300.jsonl" `
        --out "results/gpu_quick_pubmedqa_6b" `
        --limit 40 --cells "H1,H4,H5,SYC" --resume
} -ArgumentList $Device6B, $wd

$job7 = Start-Job -Name "quick-7b" -ScriptBlock {
    param($dev, $wd)
    Set-Location $wd
    $env:CUDA_VISIBLE_DEVICES = $dev
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    python -m bts_audit.cli run `
        --config "configs/gpu_quick_pubmedqa_7b.yaml" `
        --data "data/cases/pubmedqa_300.jsonl" `
        --out "results/gpu_quick_pubmedqa_7b" `
        --limit 40 --cells "H1,H4,H5,SYC" --resume
} -ArgumentList $Device7B, $wd

Write-Host "Both probes started. Check progress with: Get-Job | Select Name,State" -ForegroundColor Cyan
$jobs = @($job6, $job7)
$jobs | Wait-Job | Out-Null

foreach ($j in $jobs) {
    Write-Host "===== $($j.Name): $($j.State) =====" -ForegroundColor Green
    Receive-Job $j
    Remove-Job $j
}
