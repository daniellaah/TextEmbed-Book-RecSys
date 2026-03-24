#!/usr/bin/env python3
"""Compare one embedding eval run against baseline reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
DEFAULT_XDG_CACHE_HOME = REPO_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_XDG_CACHE_HOME))
DEFAULT_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
DEFAULT_IMG_DIR = REPO_ROOT / "img"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baselines vs one embedding eval run.")
    parser.add_argument(
        "--embedding-eval-dir",
        required=True,
        help="Embedding eval directory containing run_eval_report.json.",
    )
    parser.add_argument(
        "--baseline-root",
        default="outputs/baselines",
        help="Root directory containing baseline outputs.",
    )
    parser.add_argument(
        "--baselines",
        default="tfidf,category_popular,global_popular,category_random,random",
        help="Comma-separated baseline names to include.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to img/baseline_comparison.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def load_embedding_report(eval_dir: Path) -> dict[str, Any]:
    report_path = eval_dir / "run_eval_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing embedding report: {report_path}")
    report = load_json(report_path)
    config = report.get("config")
    if not isinstance(config, dict):
        config = {}
    eval_stats = report.get("eval_stats")
    if not isinstance(eval_stats, dict):
        eval_stats = {}
    embedding_identity = report.get("embedding_identity")
    if not isinstance(embedding_identity, dict):
        embedding_identity = {}
    index_stats = report.get("index_stats")
    if not isinstance(index_stats, dict):
        index_stats = {}
    model_name = normalize_text(embedding_identity.get("model_name_guess")) or "embedding"
    embedding_dim = normalize_text(index_stats.get("embedding_dim"))
    label = model_name if not embedding_dim else f"{model_name} ({embedding_dim})"
    return {
        "name": label,
        "kind": "embedding",
        "report_path": str(report_path.resolve()),
        "metrics": report.get("metrics", {}),
        "eval_input": normalize_text(config.get("eval_input")),
        "topk": config.get("topk", []),
        "max_query": int(config.get("max_query", 0) or 0),
        "valid_eval_rows": int(eval_stats.get("valid_eval_rows", 0)),
    }


def resolve_latest_baseline_report(baseline_root: Path, baseline_name: str) -> Path:
    baseline_dir = baseline_root / baseline_name
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Missing baseline directory: {baseline_dir}")
    report_paths = sorted(
        baseline_dir.glob("*/report.json"),
        key=lambda path: path.parent.name,
    )
    if not report_paths:
        raise FileNotFoundError(f"No report.json found under {baseline_dir}")
    return report_paths[-1]


def load_baseline_report(baseline_root: Path, baseline_name: str) -> dict[str, Any]:
    report_path = resolve_latest_baseline_report(baseline_root, baseline_name)
    report = load_json(report_path)
    config = report.get("config")
    if not isinstance(config, dict):
        config = {}
    eval_stats = report.get("eval_stats")
    if not isinstance(eval_stats, dict):
        eval_stats = {}
    return {
        "name": baseline_name,
        "kind": "baseline",
        "report_path": str(report_path.resolve()),
        "metrics": report.get("metrics", {}),
        "eval_input": normalize_text(config.get("eval_input")),
        "topk": config.get("topk", []),
        "max_query": int(config.get("max_query", 0) or 0),
        "valid_eval_rows": int(eval_stats.get("valid_eval_rows", 0)),
    }


def collect_systems(embedding_eval_dir: Path, baseline_root: Path, baseline_names: list[str]) -> list[dict[str, Any]]:
    systems = [load_embedding_report(embedding_eval_dir)]
    for baseline_name in baseline_names:
        systems.append(load_baseline_report(baseline_root, baseline_name))
    return systems


def validate_system_compatibility(systems: list[dict[str, Any]]) -> None:
    if not systems:
        raise ValueError("No systems to compare.")
    reference = systems[0]
    reference_eval_input = normalize_text(reference.get("eval_input"))
    reference_topk = [int(x) for x in reference.get("topk", [])]
    reference_max_query = int(reference.get("max_query", 0))
    reference_valid_eval_rows = int(reference.get("valid_eval_rows", 0))

    for system in systems[1:]:
        system_eval_input = normalize_text(system.get("eval_input"))
        system_topk = [int(x) for x in system.get("topk", [])]
        system_max_query = int(system.get("max_query", 0))
        system_valid_eval_rows = int(system.get("valid_eval_rows", 0))
        if system_eval_input != reference_eval_input:
            raise ValueError(
                f"Incompatible eval_input between {reference['name']} and {system['name']}: "
                f"{reference_eval_input} != {system_eval_input}"
            )
        if system_topk != reference_topk:
            raise ValueError(
                f"Incompatible topk between {reference['name']} and {system['name']}: "
                f"{reference_topk} != {system_topk}"
            )
        if system_max_query != reference_max_query:
            raise ValueError(
                f"Incompatible max_query between {reference['name']} and {system['name']}: "
                f"{reference_max_query} != {system_max_query}"
            )
        if system_valid_eval_rows != reference_valid_eval_rows:
            raise ValueError(
                f"Incompatible valid_eval_rows between {reference['name']} and {system['name']}: "
                f"{reference_valid_eval_rows} != {system_valid_eval_rows}"
            )


def build_records(systems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for system in systems:
        metrics = system.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for at_k, metric_values in metrics.items():
            if not isinstance(metric_values, dict):
                continue
            k = normalize_text(at_k).lstrip("@")
            for metric_name, metric_value in metric_values.items():
                records.append(
                    {
                        "system_name": system["name"],
                        "system_kind": system["kind"],
                        "report_path": system["report_path"],
                        "metric_name": normalize_text(metric_name),
                        "k": k,
                        "metric_key": f"{normalize_text(metric_name)}@{k}",
                        "metric_value": float(metric_value),
                    }
                )
    return records


def write_results_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "system_name",
        "system_kind",
        "report_path",
        "metric_name",
        "k",
        "metric_key",
        "metric_value",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda x: (x["metric_name"], int(x["k"]), x["system_name"])):
            writer.writerow(record)


def write_summary(
    systems: list[dict[str, Any]],
    output_path: Path,
    embedding_eval_dir: Path,
    baseline_root: Path,
) -> None:
    payload = {
        "embedding_eval_dir": str(embedding_eval_dir.resolve()),
        "baseline_root": str(baseline_root.resolve()),
        "systems": [
            {
                "name": system["name"],
                "kind": system["kind"],
                "report_path": system["report_path"],
            }
            for system in systems
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_series(records: list[dict[str, Any]], metric_name: str, ks: list[str]) -> dict[str, list[float]]:
    values_by_system: dict[str, dict[str, float]] = {}
    for record in records:
        if record["metric_name"] != metric_name:
            continue
        values_by_system.setdefault(record["system_name"], {})[record["k"]] = float(record["metric_value"])
    series: dict[str, list[float]] = {}
    for system_name, k_to_value in values_by_system.items():
        missing_ks = [k for k in ks if k not in k_to_value]
        if missing_ks:
            raise ValueError(
                f"Missing {metric_name} values for system {system_name} at K={','.join(missing_ks)}"
            )
        series[system_name] = [k_to_value[k] for k in ks]
    return series


def plot_metric_family(
    *,
    metric_name: str,
    systems: list[dict[str, Any]],
    records: list[dict[str, Any]],
    ks: list[str],
    output_path: Path,
) -> None:
    series = build_series(records, metric_name, ks)
    if not series:
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    x_positions = np.arange(len(ks), dtype=float)
    all_values = [value for values in series.values() for value in values]
    num_systems = len(systems)
    bar_width = min(0.8 / max(num_systems, 1), 0.22)
    offsets = (np.arange(num_systems, dtype=float) - (num_systems - 1) / 2.0) * bar_width

    for idx, system in enumerate(systems):
        name = system["name"]
        values = series.get(name)
        if values is None:
            continue
        is_embedding = system["kind"] == "embedding"
        ax.bar(
            x_positions + offsets[idx],
            values,
            width=bar_width * 0.92,
            label=name,
            alpha=1.0 if is_embedding else 0.9,
            edgecolor="black" if is_embedding else None,
            linewidth=1.0 if is_embedding else 0.0,
        )

    ax.set_title(f"{metric_name.upper()} Comparison")
    ax.set_xlabel("K")
    ax.set_ylabel(metric_name.upper())
    ax.set_xticks(x_positions, [f"@{k}" for k in ks])
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        pad = max(abs(min_value) * 0.05, 1e-4)
    else:
        pad = max((max_value - min_value) * 0.15, 1e-4)
    lower = max(min_value - pad, 0.0) if min_value >= 0.0 else min_value - pad
    upper = min(max_value + pad, 1.0) if max_value <= 1.0 else max_value + pad
    if lower >= upper:
        upper = lower + 1e-3
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    embedding_eval_dir = Path(args.embedding_eval_dir).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    baseline_names = [x.strip() for x in args.baselines.split(",") if x.strip()]
    if not baseline_names:
        raise ValueError("--baselines cannot be empty")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else DEFAULT_IMG_DIR / "baseline_comparison"
    )
    ensure_dir(output_dir)

    systems = collect_systems(embedding_eval_dir, baseline_root, baseline_names)
    validate_system_compatibility(systems)
    records = build_records(systems)
    if not records:
        raise ValueError("No metric records collected.")

    write_results_csv(records, output_dir / "results.csv")
    write_summary(systems, output_dir / "summary.json", embedding_eval_dir, baseline_root)

    ks = sorted({record["k"] for record in records}, key=lambda x: int(x))
    for metric_name in ("recall", "mrr", "ndcg"):
        plot_metric_family(
            metric_name=metric_name,
            systems=systems,
            records=records,
            ks=ks,
            output_path=output_dir / f"{metric_name}_comparison.png",
        )

    print(f"[compare] embedding_eval_dir={embedding_eval_dir}")
    print(f"[compare] baseline_root={baseline_root}")
    print(f"[compare] output_dir={output_dir}")
    print(f"[compare] systems={len(systems)}")


if __name__ == "__main__":
    main()
