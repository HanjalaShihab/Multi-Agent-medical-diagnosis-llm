"""Command line interface.

Examples
--------
    python -m bts_audit.cli prepare --dataset medqa --raw data/raw/medqa \
        --out data/cases/medqa_300.jsonl --n 300
    python -m bts_audit.cli run --config configs/pilot_medqa.yaml
    python -m bts_audit.cli analyze --run results/medqa_300
    python -m bts_audit.cli doctor
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import urllib.request
from pathlib import Path
from typing import Any

from .analysis import compute_analysis
from .config import DEFAULTS, get, load_config, save_config
from .data.loaders import dataset_stats, load_cases, prepare_dataset
from .runner import ExperimentRunner
from .team import CELL_DESCRIPTIONS
from .utils import write_json


def _project_root(config_path: Path) -> Path:
    # configs live in <project>/configs/*.yaml
    return config_path.resolve().parent.parent


def _resolve(path_value: str | None, config_path: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = _project_root(config_path) / path
    return candidate


def cmd_prepare(args: argparse.Namespace) -> int:
    field_map = None
    if args.field_map:
        if Path(args.field_map).exists():
            field_map = json.loads(Path(args.field_map).read_text(encoding="utf-8"))
        else:
            field_map = json.loads(args.field_map)
    out = Path(args.out) if args.out else Path("data/cases") / f"{args.dataset}.jsonl"
    cases = prepare_dataset(
        dataset=args.dataset,
        raw_path=args.raw,
        out_path=out,
        n=args.n,
        seed=args.seed,
        max_options=args.max_options,
        field_map=field_map,
        subjects=args.subjects,
    )
    stats = dataset_stats(cases)
    stats_path = out.with_suffix(".stats.json")
    write_json(stats_path, stats)
    print(f"Wrote {len(cases)} cases to {out}")
    print(f"Stats: {json.dumps(stats['strata'], sort_keys=True)}")
    print(f"Sources: {json.dumps(stats['sources'], sort_keys=True)}")
    print(f"Stats file: {stats_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    cfg = load_config(config_path)
    if args.backend:
        cfg["backend"]["type"] = args.backend
    if args.cells:
        cfg["cells"] = [c.strip().upper() for c in args.cells.split(",") if c.strip()]
    if args.limit:
        cfg["data"]["limit"] = args.limit
    if args.data:
        cfg["data"]["path"] = args.data
    if args.out:
        cfg["output_dir"] = args.out

    data_path = _resolve(cfg["data"].get("path"), config_path)
    if data_path is None or not Path(data_path).exists():
        raise FileNotFoundError(
            f"case file not found: {data_path}. "
            "Run `python -m bts_audit.cli prepare ...` first."
        )

    cases = load_cases(data_path)
    out_dir = _resolve(cfg.get("output_dir"), config_path) or (_project_root(config_path) / "results" / cfg["run_name"])
    ensure_run_dir = Path(out_dir)
    ensure_run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, ensure_run_dir / "config_used.yaml")

    limit = cfg["data"].get("limit")
    cases_shown = f"{limit} of {len(cases)}" if limit else str(len(cases))
    print(
        f"Backend: {cfg['backend']['type']} | cases: {cases_shown} | cells: {cfg['cells']}"
    )
    runner = ExperimentRunner(cfg, cases, ensure_run_dir)
    runner.run(cells=cfg["cells"], limit=cfg["data"].get("limit"), resume=args.resume)

    if not args.no_analyze:
        result = compute_analysis(ensure_run_dir, cfg=cfg)
        print(f"\nReport: {result['report']}")
        print(f"Verdict: {result['predictions'].get('verdict')}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    cfg = load_config(args.config) if args.config else None
    result = compute_analysis(run_dir, cfg=cfg, variant=args.variant)
    print(f"Report: {result['report']}")
    print(f"Verdict: {result['predictions'].get('verdict')}")
    metrics = result["metrics"]
    print("\nCell   dAUC    AUC    rho    rare-premium   oracle-dAUC")
    for cell, m in metrics.items():
        print(
            f"{cell:<5} {_f(m.get('delta_auc')):<7} {_f(m.get('auc_score')):<6} "
            f"{_f(m.get('rho')):<6} {_f(m.get('rarity_premium')):<14} {_f(m.get('delta_auc_oracle'))}"
        )
    return 0


def _f(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if out != out else f"{out:.3f}"


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
    in_kaggle = bool(Path("/kaggle/working").exists() or __import__("os").environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    if in_kaggle:
        print("  environment: Kaggle")
    for name in ("numpy", "pandas", "yaml", "tqdm", "pyarrow"):
        try:
            module = __import__(name)
            print(f"  {name}: {getattr(module, '__version__', 'ok')}")
        except Exception as exc:
            print(f"  {name}: MISSING ({exc})")
    try:
        import torch

        print(f"  torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            print(f"  GPU: {', '.join(names)}")
    except Exception as exc:
        print(f"  torch: not installed ({exc})")
    try:
        import transformers

        print(f"  transformers: {transformers.__version__}")
    except Exception as exc:
        print(f"  transformers: not installed ({exc})")
    try:
        import bitsandbytes as bnb

        print(f"  bitsandbytes: {getattr(bnb, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  bitsandbytes: not installed ({exc})")

    if args.config:
        cfg = load_config(args.config)
        print(f"Config OK: {args.config} | backend={cfg['backend']['type']} | cells={cfg['cells']}")
        data_path = _resolve(cfg["data"].get("path"), Path(args.config))
        print(f"Data path: {data_path} | exists={bool(data_path and Path(data_path).exists())}")
        out_path = _resolve(cfg.get("output_dir"), Path(args.config))
        if out_path:
            print(f"Output dir: {out_path}")
        if cfg["backend"]["type"] == "ollama":
            host = cfg["backend"]["ollama"].get("host", "http://localhost:11434")
            try:
                with urllib.request.urlopen(host + "/api/tags", timeout=3) as response:
                    tags = json.loads(response.read().decode("utf-8"))
                print(f"Ollama OK: {len(tags.get('models', []))} models")
            except Exception as exc:
                print(f"Ollama not reachable: {exc}")
    return 0


def cmd_list_cells(args: argparse.Namespace) -> int:
    for cell, desc in CELL_DESCRIPTIONS.items():
        print(f"{cell}: {desc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bts-audit",
        description="Audit Bayesian Truth Serum in multi-agent medical diagnosis. Local, no API key.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="convert a dataset into case JSONL")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["medqa", "pubmedqa", "mmlu_med", "medmcqa", "medxpertqa", "generic"],
        help="human-written benchmark to convert; `generic` is an optional escape hatch",
    )
    p.add_argument("--raw", default=None, help="raw dataset directory or file")
    p.add_argument("--out", default=None)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--max-options", type=int, default=10)
    p.add_argument("--field-map", default=None, help="JSON string or JSON file for the generic loader")
    p.add_argument(
        "--subjects",
        default=None,
        help=(
            "comma-separated MMLU subjects (default: anatomy, clinical_knowledge, "
            "college_medicine, medical_genetics, professional_medicine, virology)"
        ),
    )
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="run the H0-H5 ladder")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="output directory (overrides config output_dir)")
    p.add_argument("--data", default=None, help="case JSONL path (overrides config data.path)")
    p.add_argument("--cells", default=None, help="comma-separated subset, e.g. H1,H4")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--backend", default=None, choices=["mock", "hf", "ollama"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-analyze", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="score a finished run and write report.md")
    p.add_argument("--run", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--variant", default=None, choices=["geometric", "arithmetic"])
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("doctor", help="check environment and config")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("list-cells", help="print the experimental ladder")
    p.set_defaults(func=cmd_list_cells)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
