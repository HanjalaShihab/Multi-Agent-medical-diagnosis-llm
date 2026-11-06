#!/usr/bin/env python3
"""Download the human-written public benchmarks used by the pilot.

Everything runs locally and needs no API key. The script uses public
Hugging Face ``resolve`` URLs, so ``huggingface_hub`` is not required.

Examples
--------
    python scripts/download_data.py medqa --out data/raw/medqa
    python scripts/download_data.py pubmedqa --out data/raw/pubmedqa
    python scripts/download_data.py mmlu --out data/raw/mmlu
    python scripts/download_data.py medmcqa --out data/raw/medmcqa
    python scripts/download_data.py medxpertqa --out data/raw/medxpertqa
    python scripts/download_data.py all --out data/raw

Notes
-----
* PubMedQA uses ``pqa_labeled`` (1,000 human-annotated rows), never
  ``pqa_artificial``.
* MedMCQA uses the validation split. The official test split ships
  ``cop == -1`` for every row and cannot be scored locally.
* BioASQ is not included: the public mirrors are gated or do not ship the
  benchmark file. Treat it as optional and manual.

Review each dataset's license before redistributing derived cases.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets"

MMLU_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "virology",
]

DATASETS: dict[str, dict] = {
    "medqa": {
        "files": [
            (
                f"{HF}/GBaker/MedQA-USMLE-4-options/resolve/main/phrases_no_exclude_test.jsonl",
                "test.jsonl",
            ),
        ],
        "license": "CC-BY-4.0",
        "citation": "Jin et al., 2021 (MedQA)",
    },
    "pubmedqa": {
        "files": [
            (
                f"{HF}/qiaojin/PubMedQA/resolve/main/pqa_labeled/train-00000-of-00001.parquet",
                "pqa_labeled.parquet",
            ),
        ],
        "license": "MIT",
        "citation": "Jin et al., 2019 (PubMedQA, pqa_labeled)",
    },
    "mmlu": {
        "files": [
            (
                f"{HF}/cais/mmlu/resolve/main/{subject}/test-00000-of-00001.parquet",
                f"{subject}_test.parquet",
            )
            for subject in MMLU_SUBJECTS
        ],
        "license": "MIT",
        "citation": "Hendrycks et al., 2021 (MMLU medical subjects)",
    },
    "medmcqa": {
        "files": [
            (
                f"{HF}/openlifescienceai/medmcqa/resolve/main/data/validation-00000-of-00001.parquet",
                "validation.parquet",
            ),
        ],
        "license": "Apache-2.0",
        "citation": "Pal et al., 2022 (MedMCQA validation split)",
    },
    "medxpertqa": {
        "files": [
            (
                f"{HF}/TsinghuaC3I/MedXpertQA/resolve/main/Text/test.jsonl",
                "test.jsonl",
            ),
        ],
        "license": "MIT",
        "citation": "Zuo et al., 2025 (MedXpertQA text split)",
    },
}

MANUAL = {
    "bioasq": "BioASQ is gated/not mirrored publicly; download it manually from https://bioasq.org/ if you have access.",
}


def download(url: str, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"  exists: {out}")
        return True
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        print(f"  {url}")
        print(f"  -> {out}")
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(out)
        return True
    except Exception as exc:
        print(f"  failed: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS) + ["all", "bioasq"])
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    if args.dataset == "bioasq":
        print(MANUAL["bioasq"])
        return 1

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    root = Path(args.out)
    ok = True
    for name in names:
        spec = DATASETS[name]
        out_dir = root / name
        print(f"\n== {name} ({spec['license']}; {spec['citation']}) ==")
        for url, relative in spec["files"]:
            if not download(url, out_dir / relative):
                ok = False
        print(f"  ready under {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
