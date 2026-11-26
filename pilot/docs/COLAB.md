# Google Colab — small models on all benchmarks

This setup runs the **same small model team used on the local PC** — `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` (cycled as `0.5, 1.5, 0.5`) — across all five human-written benchmarks:

- PubMedQA (`pqa_labeled`, genuine H4 evidence partition)
- MedQA
- MMLU-Med
- MedMCQA (validation)
- MedXpertQA (text)

Use `notebooks/colab_small_models.ipynb`. It clones the repo, installs GPU deps, prepares data, runs all datasets with `--resume`, plots figures, and can zip/download results.

## Before starting Colab

On your laptop, push the new configs/script/notebook/doc:

```bash
cd "/home/hanjala-shihab/Documents/Multi-Agent medical diagnosis llm"
git add pilot/configs/colab_*_small.yaml pilot/scripts/run_colab_all_small.sh \
  pilot/notebooks/colab_small_models.ipynb pilot/docs/COLAB.md
git commit -m "Add Google Colab small-model all-dataset sweep"
git push origin main
```

Do not use `git add -A` here: the repository root now contains large untracked PDFs and the DIU Word template/guidelines, which do not need to be pushed for Colab.

## Colab settings

- Runtime → Change runtime type → **GPU (T4)**
- No API key is needed; models download from public Hugging Face.

## Notebook flow

1. Clone/pull repo into `/content/Multi-Agent-medical-diagnosis-llm`.
2. Check CUDA with `nvidia-smi` / `torch.cuda.is_available()`.
3. Install `requirements-gpu.txt` and `pip install -e . --no-deps`.
4. Download raw data and prepare the five `*_300.jsonl` case files.
5. Mount Google Drive in the notebook and set `OUT_ROOT`. This keeps `answers.jsonl`/`forecasts.jsonl` outside the ephemeral `/content`, so resume survives session resets.
6. Optional quick first pass:
   ```bash
   OUT_ROOT=/content/drive/MyDrive/bts_audit_colab_results LIMIT=60 CELLS=H1,H4,H5,SYC RESUME=1 bash scripts/run_colab_all_small.sh
   ```
7. Full sweep:
   ```bash
   OUT_ROOT=/content/drive/MyDrive/bts_audit_colab_results LIMIT=300 CELLS=H0,H1,H2,H3,H4,H5,SYC RESUME=1 bash scripts/run_colab_all_small.sh
   ```
8. Results are under `${OUT_ROOT}/colab_<dataset>_small/` with `report.md`, `cell_metrics.csv`, `metrics.json`, `predictions.json`, `agent_scores.csv`, raw `answers.jsonl`/`forecasts.jsonl`, and `figures/*.png`.

## Resume behaviour

Each dataset has its **own config and output dir**. If the Colab session disconnects or the VM restarts, re-run the clone/install/data cells and the full-sweep cell with the **same `OUT_ROOT`**; `--resume` continues from the last completed `(cell, case_id, round, agent_id, task)` record. Without Google Drive, `/content` is wiped on reset and the run starts over. Do not change a config/model and resume into the same output dir.

## Notes

- The small model team is for pipeline and relative comparison, not for 90%+ diagnostic accuracy.
- PubMedQA is the only genuine H4 evidence partition; H4/H5 on the other four datasets are text-chunk ablations.
- `rarity_stratum` and `prevalence` remain `None`; P6a/P6b are mechanical checks, not disease-prevalence claims.
- To keep results after the session, download the zip produced at the end of the notebook, or mount Google Drive and copy `results/` there.
