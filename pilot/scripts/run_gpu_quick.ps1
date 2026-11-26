# Time-limited big-model quick probe (deadline rescue).
# 60 PubMedQA cases x H1,H4,H5,SYC, 120 new tokens, 4-bit, offline HF.
# Run from the pilot/ directory. Each model has its own config + output dir.
param(
    [string[]]$Models = @("4b", "6b", "7b"),
    [string]$Device = "0"
)

$env:CUDA_VISIBLE_DEVICES = $Device
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

foreach ($m in $Models) {
    Write-Host "GPU $Device quick run: $m" -ForegroundColor Green
    python -m bts_audit.cli run `
        --config "configs/gpu_quick_pubmedqa_${m}.yaml" `
        --data "data/cases/pubmedqa_300.jsonl" `
        --out "results/gpu_quick_pubmedqa_${m}" `
        --limit 60 `
        --cells "H1,H4,H5,SYC" `
        --resume
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$m failed. Stopping queue." -ForegroundColor Red
        break
    }
}
