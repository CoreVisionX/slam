"""
Run multiple VIO sequences and evaluate each with evo_ape, printing results tables.

Example:
  pixi run python tests/vio_eval_euroc.py \
    --config euroc/experiments/reproj_gating/no_reproj_gating.yaml \
    --config euroc/experiments/reproj_gating/reproj_gating.yaml \
    --eval VI_01_easy:euroc/VI_01_easy.yaml:tests/results/VI_01_easy_gt.txt \
    --eval VI_02_medium:euroc/VI_02_medium.yaml:tests/results/VI_02_medium_gt.txt
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sh
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class EvalSpec:
    name: str
    sequence_yaml: str
    gt_tum: str


@dataclass(frozen=True)
class ApeTranslationStats:
    max: float
    mean: float
    median: float
    min: float
    rmse: float
    sse: float
    std: float


@dataclass(frozen=True)
class EvalResult:
    name: str
    sequence_yaml: str
    gt_tum: str
    est_tum: str
    frames: Optional[int]
    distance_m: Optional[float]
    ape: ApeTranslationStats


# --- Regex parsers ---
_SAVED_TUM_RE = re.compile(r"Saved TUM trajectory:\s*(.+?)\s*$", re.MULTILINE)
_DISTANCE_RE = re.compile(r"Estimated Distance traveled:\s*([0-9]*\.?[0-9]+)\s*m", re.MULTILINE)
_FRAMES_RE = re.compile(r"Processing\s+(\d+)\s+frames\.\.\.", re.MULTILINE)

_METRIC_LINE_RE = re.compile(
    r"^\s*(max|mean|median|min|rmse|sse|std)\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def parse_eval_arg(raw: str) -> EvalSpec:
    """
    --eval format: NAME:SEQUENCE_YAML:GT_TUM
    """
    parts = raw.split(":", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise ValueError(f"Invalid --eval '{raw}'. Expected NAME:SEQUENCE_YAML:GT_TUM")
    name, seq, gt = (p.strip() for p in parts)
    return EvalSpec(name=name, sequence_yaml=seq, gt_tum=gt)


def run_vio(
    vio_script: str,
    sequence_yaml: str,
    config: Optional[str],
    extra_vio_args: List[str],
) -> str:
    """
    Runs:
      pixi run python <vio_script> --sequence <sequence_yaml> [--config <config>] [extra_vio_args...]
    Returns combined output (stdout+stderr).
    """
    cmd = ["run", "python", vio_script, "--sequence", sequence_yaml]
    if config is not None:
        cmd += ["--config", config]
    cmd += extra_vio_args
    out = sh.pixi(*cmd, _err_to_out=True, _tty_out=False)
    return str(out)


def extract_saved_tum_path(vio_output: str) -> str:
    m = _SAVED_TUM_RE.search(vio_output)
    if not m:
        raise RuntimeError(
            "VIO did not print 'Saved TUM trajectory: ...' — cannot locate estimated trajectory path.\n"
            "Ensure your vio script prints that line, or that it wasn't suppressed."
        )
    return m.group(1).strip()


def extract_distance_m(vio_output: str) -> Optional[float]:
    m = _DISTANCE_RE.search(vio_output)
    return float(m.group(1)) if m else None


def extract_frames(vio_output: str) -> Optional[int]:
    m = _FRAMES_RE.search(vio_output)
    return int(m.group(1)) if m else None


def run_evo_ape(est_tum: str, gt_tum: str) -> str:
    """
    Runs:
      uvx --with rerun-sdk --from evo evo_ape tum <est> <gt> --align
    Returns combined output (stdout+stderr).
    """
    out = sh.uvx(
        "--with",
        "rerun-sdk",
        "--from",
        "evo",
        "evo_ape",
        "tum",
        est_tum,
        gt_tum,
        "--align",
        _err_to_out=True,
        _tty_out=False,
    )
    return str(out)


def parse_ape_translation_stats(evo_output: str) -> ApeTranslationStats:
    metrics: Dict[str, float] = {}
    for line in evo_output.splitlines():
        m = _METRIC_LINE_RE.match(line)
        if m:
            metrics[m.group(1)] = float(m.group(2))

    required = ("max", "mean", "median", "min", "rmse", "sse", "std")
    missing = [k for k in required if k not in metrics]
    if missing:
        tail = "\n".join(evo_output.splitlines()[-60:])
        raise RuntimeError(
            f"Failed to parse evo_ape APE translation metrics. Missing: {missing}\n\n"
            f"Last 60 lines of evo output:\n{tail}"
        )

    return ApeTranslationStats(**{k: metrics[k] for k in required})  # type: ignore[arg-type]


def _fmt_float(x: Optional[float], decimals: int) -> str:
    if x is None:
        return ""
    return f"{x:.{decimals}f}"


def _fmt_int(x: Optional[int]) -> str:
    return "" if x is None else str(x)


def _avg(values: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _avg_int(values: List[Optional[int]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _pct_err(rmse_m: float, dist_m: Optional[float]) -> Optional[float]:
    if dist_m is None or dist_m == 0:
        return None
    return (rmse_m / dist_m) * 100.0


def print_table(rows: List[EvalResult], decimals: int = 2, title: str = "VIO ATE/APE Results") -> None:
    console = Console()
    table = Table(title=title, show_lines=False)

    table.add_column("name", no_wrap=True)
    table.add_column("sequence", overflow="fold")
    table.add_column("frames", justify="right")
    table.add_column("dist_m", justify="right")
    table.add_column("rmse_m", justify="right")
    table.add_column("pct_err", justify="right")
    table.add_column("mean_m", justify="right")
    table.add_column("median_m", justify="right")
    table.add_column("std_m", justify="right")
    table.add_column("min_m", justify="right")
    table.add_column("max_m", justify="right")
    table.add_column("sse", justify="right")
    table.add_column("est_tum", overflow="fold")
    table.add_column("gt_tum", overflow="fold")

    for r in rows:
        pe = _pct_err(r.ape.rmse, r.distance_m)
        table.add_row(
            r.name,
            r.sequence_yaml,
            _fmt_int(r.frames),
            _fmt_float(r.distance_m, decimals),
            _fmt_float(r.ape.rmse, decimals),
            ("" if pe is None else f"{pe:.{decimals}f}%"),
            _fmt_float(r.ape.mean, decimals),
            _fmt_float(r.ape.median, decimals),
            _fmt_float(r.ape.std, decimals),
            _fmt_float(r.ape.min, decimals),
            _fmt_float(r.ape.max, decimals),
            _fmt_float(r.ape.sse, decimals),
            r.est_tum,
            r.gt_tum,
        )

    avg_frames = _avg_int([r.frames for r in rows])
    avg_dist = _avg([r.distance_m for r in rows])

    avg_rmse = _avg([r.ape.rmse for r in rows])
    avg_mean = _avg([r.ape.mean for r in rows])
    avg_median = _avg([r.ape.median for r in rows])
    avg_std = _avg([r.ape.std for r in rows])
    avg_min = _avg([r.ape.min for r in rows])
    avg_max = _avg([r.ape.max for r in rows])
    avg_sse = _avg([r.ape.sse for r in rows])
    avg_pct_err = _avg([_pct_err(r.ape.rmse, r.distance_m) for r in rows])

    table.add_section()
    table.add_row(
        "[b]AVERAGE[/b]",
        "",
        _fmt_float(avg_frames, decimals),
        _fmt_float(avg_dist, decimals),
        _fmt_float(avg_rmse, decimals),
        ("" if avg_pct_err is None else f"{avg_pct_err:.{decimals}f}%"),
        _fmt_float(avg_mean, decimals),
        _fmt_float(avg_median, decimals),
        _fmt_float(avg_std, decimals),
        _fmt_float(avg_min, decimals),
        _fmt_float(avg_max, decimals),
        _fmt_float(avg_sse, decimals),
        "",
        "",
    )

    console.print(table)


def print_config_vs_traj_table(
    specs: List[EvalSpec],
    configs: List[Optional[str]],
    results_map: Dict[Tuple[str, str], EvalResult],
    decimals: int,
) -> None:
    console = Console()
    table = Table(title="Config × Trajectory Summary (rmse_m / pct_err)", show_lines=False)

    table.add_column("config", no_wrap=True)
    for spec in specs:
        table.add_column(spec.name, justify="right")

    def cfg_label(c: Optional[str]) -> str:
        return "(default)" if c is None else c

    for cfg in configs:
        row: List[str] = [cfg_label(cfg)]
        for spec in specs:
            r = results_map.get((cfg_label(cfg), spec.name))
            if r is None:
                row.append("")
                continue
            pe = _pct_err(r.ape.rmse, r.distance_m)
            rmse_str = _fmt_float(r.ape.rmse, decimals)
            pe_str = "" if pe is None else f"{pe:.{decimals}f}%"
            row.append(f"{rmse_str} / {pe_str}" if pe_str else f"{rmse_str} /")
        table.add_row(*row)

    console.print(table)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vio-script", default="tests/vio_example_euroc.py")
    ap.add_argument(
        "--eval",
        action="append",
        required=True,
        help="Repeatable. Format: NAME:SEQUENCE_YAML:GT_TUM",
    )
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        help="Repeatable. Path/name passed to VIO as --config <value>. If omitted, runs once with no --config.",
    )
    ap.add_argument(
        "--vio-arg",
        action="append",
        default=[],
        help="Extra args forwarded to the VIO script (repeatable).",
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places for numeric columns (default: 2).",
    )
    args = ap.parse_args()

    console = Console()

    specs = [parse_eval_arg(e) for e in args.eval]
    configs: List[Optional[str]] = [None] if len(args.config) == 0 else list(args.config)

    # key: (config_label, traj_name) -> EvalResult
    results_map: Dict[Tuple[str, str], EvalResult] = {}
    successful_configs: List[Optional[str]] = []

    def cfg_label(c: Optional[str]) -> str:
        return "(default)" if c is None else c

    for cfg in configs:
        cfg_results: List[EvalResult] = []
        cfg_ok = True

        console.print(f"\n[bold]Running config:[/bold] {cfg_label(cfg)}")

        # We only commit results for this config if *all* subprocesses succeed.
        try:
            for spec in specs:
                vio_out = run_vio(args.vio_script, spec.sequence_yaml, cfg, args.vio_arg)

                est_tum = extract_saved_tum_path(vio_out)  # REQUIRED
                frames = extract_frames(vio_out)
                dist_m = extract_distance_m(vio_out)

                evo_out = run_evo_ape(est_tum, spec.gt_tum)
                ape = parse_ape_translation_stats(evo_out)

                cfg_results.append(
                    EvalResult(
                        name=spec.name,
                        sequence_yaml=spec.sequence_yaml,
                        gt_tum=spec.gt_tum,
                        est_tum=est_tum,
                        frames=frames,
                        distance_m=dist_m,
                        ape=ape,
                    )
                )

        except (sh.ErrorReturnCode, RuntimeError) as e:
            cfg_ok = False
            console.print(f"[red]Skipping config due to failure:[/red] {cfg_label(cfg)}")
            console.print(f"[dim]{e}[/dim]")

        if not cfg_ok:
            continue

        # Commit this config's results
        successful_configs.append(cfg)
        for r in cfg_results:
            results_map[(cfg_label(cfg), r.name)] = r

        cfg_results.sort(key=lambda r: r.ape.rmse)
        print_table(
            cfg_results,
            decimals=args.decimals,
            title=f"VIO ATE/APE Results — config: {cfg_label(cfg)}",
        )

    if not successful_configs:
        console.print("\n[red]No configs completed successfully — nothing to summarize.[/red]")
        return 1

    print_config_vs_traj_table(
        specs=specs,
        configs=successful_configs,
        results_map=results_map,
        decimals=args.decimals,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())