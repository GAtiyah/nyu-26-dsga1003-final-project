"""Create publication-style figures and summary tables from saved results.

This stage is intentionally read-only with respect to experiment artifacts:
it reads JSON files from `result/metrics/` and `result/analysis/`, then writes
figures under `result/figures/` and a compact CSV table under `result/tables/`.

Example
-------
python script/5_make_plots.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    short_label: str
    filenames: tuple[str, ...]
    color: str


MODEL_SPECS = (
    ModelSpec(
        key="frozen_base",
        label="Frozen base",
        short_label="Frozen\nbase",
        filenames=("frozen_base.json",),
        color="#6E6E6E",
    ),
    ModelSpec(
        key="static_write_strength",
        label="Static write-strength",
        short_label="Static\nwrite-strength",
        filenames=("static_residual.json", "static_write_strength.json"),
        color="#0072B2",
    ),
    ModelSpec(
        key="prompt_write_strength",
        label="Prompt-conditioned write-strength",
        short_label="Prompt-cond.\nwrite-strength",
        filenames=("prompt_conditioned.json", "prompt_conditioned_write_strength.json"),
        color="#56B4E9",
    ),
    ModelSpec(
        key="static_reaggregation",
        label="Static re-aggregation",
        short_label="Static\nre-aggregation",
        filenames=("static_residual_stream_reaggregation.json",),
        color="#009E73",
    ),
    ModelSpec(
        key="prompt_reaggregation",
        label="Prompt-conditioned re-aggregation",
        short_label="Prompt-cond.\nre-aggregation",
        filenames=("prompt_conditioned_residual_stream_reaggregation.json",),
        color="#D55E00",
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Generate publication-style plots from saved project results."
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(repo_root / "result" / "metrics"),
        help="Directory containing stage-3 metrics JSON files.",
    )
    parser.add_argument(
        "--analysis-path",
        default=str(repo_root / "result" / "analysis" / "mlp_inspection.json"),
        help="Path to the stage-4 MLP inspection JSON file.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(repo_root / "result" / "figures"),
        help="Directory where generated figures will be written.",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(repo_root / "result" / "tables"),
        help="Directory where generated summary tables will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Save resolution for raster figures. Default: 300.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
        help="Figure formats to save. Default: png pdf.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_metric_path(metrics_dir: Path, spec: ModelSpec) -> Path:
    for filename in spec.filenames:
        path = metrics_dir / filename
        if path.exists():
            return path
    candidates = ", ".join(spec.filenames)
    raise FileNotFoundError(f"Missing metrics file for {spec.label}: {candidates}")


def load_all_metrics(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    metrics = {}
    for spec in MODEL_SPECS:
        metrics[spec.key] = load_json(resolve_metric_path(metrics_dir, spec))
    return metrics


def get_split_metric(
    metrics: dict[str, Any],
    split_name: str,
    metric_name: str,
) -> float:
    return float(metrics[split_name][metric_name])


def get_source_metric(
    metrics: dict[str, Any],
    split_name: str,
    source_name: str,
    metric_name: str,
) -> float:
    return float(metrics[split_name]["by_source"][source_name][metric_name])


def set_paper_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure,
    figures_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = figures_dir / f"{stem}.{extension}"
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.2f}",
    dy: float = 0.04,
    fontsize: int = 7,
) -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * dy
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def write_metrics_summary(
    metrics_by_model: dict[str, dict[str, Any]],
    tables_dir: Path,
) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "metrics_summary.csv"
    base_test_ppl = get_split_metric(
        metrics_by_model["frozen_base"],
        split_name="test",
        metric_name="perplexity",
    )

    fieldnames = [
        "model_key",
        "model",
        "validation_loss",
        "validation_perplexity",
        "test_loss",
        "test_perplexity",
        "test_wikitext_loss",
        "test_wikitext_perplexity",
        "test_medical_loss",
        "test_medical_perplexity",
        "test_ppl_reduction_percent",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for spec in MODEL_SPECS:
            metrics = metrics_by_model[spec.key]
            test_ppl = get_split_metric(metrics, "test", "perplexity")
            reduction = (base_test_ppl - test_ppl) / base_test_ppl * 100.0
            writer.writerow(
                {
                    "model_key": spec.key,
                    "model": spec.label,
                    "validation_loss": f"{get_split_metric(metrics, 'validation', 'loss'):.6f}",
                    "validation_perplexity": f"{get_split_metric(metrics, 'validation', 'perplexity'):.6f}",
                    "test_loss": f"{get_split_metric(metrics, 'test', 'loss'):.6f}",
                    "test_perplexity": f"{test_ppl:.6f}",
                    "test_wikitext_loss": f"{get_source_metric(metrics, 'test', 'wikitext', 'loss'):.6f}",
                    "test_wikitext_perplexity": f"{get_source_metric(metrics, 'test', 'wikitext', 'perplexity'):.6f}",
                    "test_medical_loss": f"{get_source_metric(metrics, 'test', 'medical', 'loss'):.6f}",
                    "test_medical_perplexity": f"{get_source_metric(metrics, 'test', 'medical', 'perplexity'):.6f}",
                    "test_ppl_reduction_percent": f"{reduction:.3f}",
                }
            )
    return output_path


def plot_test_perplexity_comparison(
    metrics_by_model: dict[str, dict[str, Any]],
    figures_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    values = [
        get_split_metric(metrics_by_model[spec.key], "test", "perplexity")
        for spec in MODEL_SPECS
    ]
    colors = [spec.color for spec in MODEL_SPECS]
    labels = [spec.short_label for spec in MODEL_SPECS]
    best_index = min(range(len(values)), key=values.__getitem__)

    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    bars = ax.bar(
        range(len(values)),
        values,
        color=colors,
        edgecolor=["#202020" if i == best_index else "#FFFFFF" for i in range(len(values))],
        linewidth=[1.3 if i == best_index else 0.7 for i in range(len(values))],
        width=0.68,
    )
    ax.set_ylabel("Test perplexity")
    ax.set_title("Test perplexity across model variants")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.17)
    set_paper_axes(ax)
    annotate_bars(ax, bars, fmt="{:.2f}", dy=0.015)
    ax.text(
        best_index,
        values[best_index] * 1.08,
        "best",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#202020",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(
        fig=fig,
        figures_dir=figures_dir,
        stem="01_test_perplexity_comparison",
        formats=formats,
        dpi=dpi,
    )


def plot_test_perplexity_by_source(
    metrics_by_model: dict[str, dict[str, Any]],
    figures_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    sources = (("wikitext", "WikiText"), ("medical", "Medical"))
    group_positions = [0.0, 1.0]
    bar_width = 0.15
    offsets = [
        (index - (len(MODEL_SPECS) - 1) / 2) * bar_width
        for index in range(len(MODEL_SPECS))
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    all_values: list[float] = []
    for model_index, spec in enumerate(MODEL_SPECS):
        values = [
            get_source_metric(metrics_by_model[spec.key], "test", source_key, "perplexity")
            for source_key, _source_label in sources
        ]
        all_values.extend(values)
        positions = [group + offsets[model_index] for group in group_positions]
        ax.bar(
            positions,
            values,
            width=bar_width * 0.92,
            color=spec.color,
            edgecolor="#FFFFFF",
            linewidth=0.6,
            label=spec.label,
        )

    ax.set_ylabel("Test perplexity")
    ax.set_title("Domain-wise test perplexity")
    ax.set_xticks(group_positions)
    ax.set_xticklabels([source_label for _source_key, source_label in sources])
    ax.set_ylim(0, max(all_values) * 1.16)
    ax.legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=1.4)
    set_paper_axes(ax)
    fig.tight_layout()
    save_figure(
        fig=fig,
        figures_dir=figures_dir,
        stem="02_test_perplexity_by_source",
        formats=formats,
        dpi=dpi,
    )


def plot_relative_improvement(
    metrics_by_model: dict[str, dict[str, Any]],
    figures_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    base_ppl = get_split_metric(metrics_by_model["frozen_base"], "test", "perplexity")
    values = []
    for spec in MODEL_SPECS:
        test_ppl = get_split_metric(metrics_by_model[spec.key], "test", "perplexity")
        values.append((base_ppl - test_ppl) / base_ppl * 100.0)

    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    bars = ax.bar(
        range(len(values)),
        values,
        color=[spec.color for spec in MODEL_SPECS],
        edgecolor="#FFFFFF",
        linewidth=0.7,
        width=0.68,
    )
    ax.axhline(0, color="#202020", linewidth=0.8)
    ax.set_ylabel("Test PPL reduction vs. frozen base (%)")
    ax.set_title("Relative improvement over the frozen base model")
    ax.set_xticks(range(len(MODEL_SPECS)))
    ax.set_xticklabels([spec.short_label for spec in MODEL_SPECS])
    ax.set_ylim(-0.7, max(values) * 1.22)
    set_paper_axes(ax)
    annotate_bars(ax, bars, fmt="{:.1f}", dy=0.018)
    fig.tight_layout()
    save_figure(
        fig=fig,
        figures_dir=figures_dir,
        stem="03_relative_improvement_over_frozen_base",
        formats=formats,
        dpi=dpi,
    )


def get_mlp_split(
    analysis: dict[str, Any],
    model_key: str,
    split_name: str = "test",
) -> dict[str, Any]:
    return analysis[model_key]["splits"][split_name]


def plot_mlp_inspection_summary(
    analysis: dict[str, Any],
    figures_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    model_keys = (
        "prompt_conditioned_write_strength",
        "prompt_conditioned_residual_stream_reaggregation",
    )
    labels = ("Write-strength", "Re-aggregation")
    colors = ("#56B4E9", "#D55E00")
    metrics = (
        ("collapse_cosine", "Collapse cosine"),
        ("domain_separation_cosine_gap", "Domain separation cosine gap"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (metric_key, title) in zip(axes, metrics):
        values = [
            float(get_mlp_split(analysis, model_key)[metric_key])
            for model_key in model_keys
        ]
        bars = ax.bar(
            range(len(values)),
            values,
            width=0.58,
            color=colors,
            edgecolor="#FFFFFF",
            linewidth=0.7,
        )
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(values) * 1.25)
        set_paper_axes(ax)
        annotate_bars(ax, bars, fmt="{:.3f}", dy=0.018, fontsize=7)

    fig.suptitle("Prompt-conditioned MLP inspection on the test split", y=1.02)
    fig.tight_layout()
    save_figure(
        fig=fig,
        figures_dir=figures_dir,
        stem="04_mlp_inspection_summary",
        formats=formats,
        dpi=dpi,
    )


def plot_per_layer_mlp_behavior(
    analysis: dict[str, Any],
    figures_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    write_split = get_mlp_split(analysis, "prompt_conditioned_write_strength")
    reagg_split = get_mlp_split(
        analysis,
        "prompt_conditioned_residual_stream_reaggregation",
    )

    layer_count = len(write_split["per_layer_variance"])
    layer_indices = list(range(layer_count))
    panels = (
        (
            "per_layer_variance",
            "Raw MLP output variance",
            "Variance",
        ),
        (
            "per_layer_domain_mean_difference",
            "Domain mean difference",
            "Mean absolute difference",
        ),
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, (metric_key, title, ylabel) in zip(axes, panels):
        write_values = [float(value) for value in write_split[metric_key]]
        reagg_values = [float(value) for value in reagg_split[metric_key]]
        ax.plot(
            layer_indices,
            write_values,
            color="#56B4E9",
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            label="Write-strength",
        )
        ax.plot(
            layer_indices,
            reagg_values,
            color="#D55E00",
            marker="s",
            markersize=3.0,
            linewidth=1.6,
            label="Re-aggregation",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        set_paper_axes(ax)

    axes[-1].set_xlabel("Decoder layer index")
    axes[-1].set_xticks(list(range(0, layer_count, 3)) + [layer_count - 1])
    axes[0].legend(loc="upper left", ncol=2)
    fig.suptitle("Per-layer prompt-conditioned MLP behavior on the test split", y=1.01)
    fig.tight_layout()
    save_figure(
        fig=fig,
        figures_dir=figures_dir,
        stem="05_per_layer_mlp_behavior",
        formats=formats,
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    metrics_dir = Path(args.metrics_dir)
    analysis_path = Path(args.analysis_path)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    formats = tuple(args.formats)

    metrics_by_model = load_all_metrics(metrics_dir)
    analysis = load_json(analysis_path)

    summary_path = write_metrics_summary(
        metrics_by_model=metrics_by_model,
        tables_dir=tables_dir,
    )
    plot_test_perplexity_comparison(
        metrics_by_model=metrics_by_model,
        figures_dir=figures_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_test_perplexity_by_source(
        metrics_by_model=metrics_by_model,
        figures_dir=figures_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_relative_improvement(
        metrics_by_model=metrics_by_model,
        figures_dir=figures_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_mlp_inspection_summary(
        analysis=analysis,
        figures_dir=figures_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_per_layer_mlp_behavior(
        analysis=analysis,
        figures_dir=figures_dir,
        formats=formats,
        dpi=args.dpi,
    )

    print(f"[done] Wrote summary table: {summary_path}")
    print(f"[done] Wrote figures to: {figures_dir}")


if __name__ == "__main__":
    main()
