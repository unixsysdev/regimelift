from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
FIG_DIR = PAPER_ROOT / "figures"
GEN_DIR = PAPER_ROOT / "generated"

TARGETED_ROWS = REPO_ROOT / "helmas3n" / "artifacts" / "reports" / "targeted_site_study_v5_holdout80" / "targeted_site_rows.csv"
LAYER_SWEEP_ROWS = REPO_ROOT / "helmas3n" / "artifacts" / "reports" / "layer_sweep_v2_corrected" / "layer_sweep_rows.csv"
SUFFIX_SPAN_ROWS = REPO_ROOT / "helmas3n" / "artifacts" / "reports" / "suffix_span_sweep_v1" / "suffix_span_summary.csv"
SANITY_REPORT = REPO_ROOT / "helmas3n" / "artifacts" / "reports" / "killtest_v3" / "sanity" / "sanity_report.json"
SANITY_LAYER_ROWS = REPO_ROOT / "helmas3n" / "artifacts" / "reports" / "killtest_v3" / "sanity" / "per_layer_metrics.csv"

TARGETED_METHOD_ORDER = [
    ("low_to_full_no_patch", "layer34", "No patch"),
    ("identity", "layer34", "Identity"),
    ("broad_mlp", "layer34", "Broad MLP"),
    ("targeted_mlp", "layer16", "Targeted MLP @ layer16"),
    ("targeted_mlp", "layer34", "Targeted MLP @ layer34"),
    ("oracle_layer16", "layer34", "Reference layer16"),
    ("oracle_layer34", "layer34", "Reference layer34"),
    ("oracle_stride", "layer34", "Reference stride"),
]

HORIZONS = [1, 4, 8, 16]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v: Any, digits: int = 6) -> str:
    if v is None or v == "":
        return "--"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
        .replace("#", r"\#").replace("$", r"\$").replace("{", r"\{").replace("}", r"\}")
        .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)


def lookup_targeted(rows: list[dict[str, str]], split: str, method: str, experiment_site: str | None = None) -> dict[str, str]:
    for row in rows:
        if row["split"] != split:
            continue
        if row["method"] != method:
            continue
        if experiment_site is not None and row.get("experiment_site") != experiment_site:
            continue
        return row
    raise KeyError((split, method, experiment_site))


def write_targeted_core_table(rows: list[dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & h1 & h4 & h8 & h16 & $\Delta$h8 & $\Delta$h16 \\")
    lines.append(r"\midrule")
    for method, site, label in TARGETED_METHOD_ORDER:
        held = lookup_targeted(rows, "heldout", method, site)
        values = [
            fmt(held[f"cont_match_h{h}_uplift_vs_full"]) for h in HORIZONS
        ] + [
            fmt(held["delta_h8"]),
            fmt(held["delta_h16"]),
        ]
        lines.append(latex_escape(label) + " & " + " & ".join(values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (GEN_DIR / "core_results_table.tex").write_text("\n".join(lines) + "\n")


def write_layer_sweep_summary_table(rows: list[dict[str, str]]) -> None:
    rows = [r for r in rows if r.get("layer", "").strip()]
    methods = ["identity", "mlp", "oracle_full_state"]
    summary = []
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        best_h8 = max(method_rows, key=lambda r: float(r["cont_match_h8_uplift_vs_full"]))
        best_h16 = max(method_rows, key=lambda r: float(r["cont_match_h16_uplift_vs_full"]))
        summary.append(
            (
                method,
                int(best_h8["layer"]),
                float(best_h8["cont_match_h8_uplift_vs_full"]),
                int(best_h16["layer"]),
                float(best_h16["cont_match_h16_uplift_vs_full"]),
            )
        )

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Best layer @ h8 & h8 & Best layer @ h16 & h16 \\")
    lines.append(r"\midrule")
    for method, layer_h8, val_h8, layer_h16, val_h16 in summary:
        lines.append(
            f"{latex_escape(method)} & {layer_h8} & {fmt(val_h8)} & {layer_h16} & {fmt(val_h16)} " + r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (GEN_DIR / "layer_sweep_summary_table.tex").write_text("\n".join(lines) + "\n")


def write_regime_separation_table(report: dict[str, Any]) -> None:
    global_metrics = report["global"]
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Value \\")
    lines.append(r"\midrule")
    rows = [
        ("Prompt count", str(global_metrics["num_prompts"])),
        ("Residual cosine mean", fmt(global_metrics["residual_cosine_mean"], 4)),
        ("Residual MSE mean", fmt(global_metrics["residual_mse_mean"], 4)),
        ("Next-token KL(full || low)", fmt(global_metrics["next_token_kl_full_to_low_mean"], 4)),
        ("Next-token top-1 agreement", fmt(global_metrics["next_token_top1_agreement_mean"], 4)),
    ]
    for label, value in rows:
        lines.append(f"{latex_escape(label)} & {latex_escape(value)} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (GEN_DIR / "regime_separation_table.tex").write_text("\n".join(lines) + "\n")


def write_regime_controls_table(report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Regime & Activation sparsity & KV shared layers & Scale correction & Active index \\")
    lines.append(r"\midrule")
    for regime_name in ["low", "full"]:
        cfg = report["regimes"][regime_name]["text_overrides"]
        lines.append(
            f"{latex_escape(regime_name)} & "
            f"{latex_escape(fmt(cfg['activation_sparsity_pattern'], 2))} & "
            f"{latex_escape(str(cfg['num_kv_shared_layers']))} & "
            f"{latex_escape(str(cfg['altup_correct_scale']))} & "
            f"{latex_escape(str(cfg['altup_active_idx']))} "
            + r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (GEN_DIR / "regime_controls_table.tex").write_text("\n".join(lines) + "\n")


def plot_targeted_curves(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.4))
    method_rows = [
        ("low_to_full_no_patch", "layer34", "No patch", "#444444", "o", "-"),
        ("identity", "layer34", "Identity", "#7f7f7f", "o", "--"),
        ("broad_mlp", "layer34", "Broad MLP", "#1f77b4", "s", "-"),
        ("targeted_mlp", "layer16", "Targeted layer16", "#ff7f0e", "^", "--"),
        ("targeted_mlp", "layer34", "Targeted layer34", "#2ca02c", "D", "-"),
        ("oracle_layer34", "layer34", "Reference layer34", "#d62728", "X", ":"),
        ("oracle_stride", "layer34", "Reference stride", "#9467bd", "P", ":"),
    ]

    for method, site, label, color, marker, ls in method_rows:
        row = lookup_targeted(rows, "heldout", method, site)
        ys = [float(row[f"cont_match_h{h}_uplift_vs_full"]) for h in HORIZONS]
        ax.plot(HORIZONS, ys, label=label, color=color, marker=marker, linewidth=2.0, linestyle=ls)

    ax.set_title("Heldout80 targeted handoff")
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Decode horizon (tokens)")
    ax.set_ylabel("Continuation match vs full")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("RegimeLift targeted handoff recovery (heldout80)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "targeted_handoff_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "targeted_handoff_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_targeted_delta_bars(rows: list[dict[str, str]]) -> None:
    method_rows = [
        ("broad_mlp", "layer34", "Broad MLP", "#1f77b4"),
        ("targeted_mlp", "layer16", "Targeted layer16", "#ff7f0e"),
        ("targeted_mlp", "layer34", "Targeted layer34", "#2ca02c"),
        ("oracle_layer34", "layer34", "Reference layer34", "#d62728"),
        ("oracle_stride", "layer34", "Reference stride", "#9467bd"),
    ]
    labels = []
    h8_vals = []
    h16_vals = []
    for method, site, label, _ in method_rows:
        row = lookup_targeted(rows, "heldout", method, site)
        labels.append(label)
        h8_vals.append(float(row["delta_h8"]))
        h16_vals.append(float(row["delta_h16"]))

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 4.6))
    ax.bar([i - width / 2 for i in x], h8_vals, width=width, label="Delta h8 vs no patch", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], h16_vals, width=width, label="Delta h16 vs no patch", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("Absolute continuation gain")
    ax.set_title("Heldout80 continuation gains relative to no patch")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "targeted_delta_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "targeted_delta_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_layer_sweep(rows: list[dict[str, str]]) -> None:
    rows = [r for r in rows if r.get("layer", "").strip()]
    layers = sorted({int(r["layer"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    configs = [
        ("cont_match_h8_uplift_vs_full", "Continuation match h8", "#1f77b4"),
        ("cont_match_h16_uplift_vs_full", "Continuation match h16", "#d62728"),
    ]
    colors = {
        "identity": "#444444",
        "mlp": "#1f77b4",
        "oracle_full_state": "#d62728",
    }
    for ax, (metric, title, color) in zip(axes, configs):
        for method, style, label in [
            ("identity", "-", "Identity"),
            ("mlp", "--", "MLP"),
            ("oracle_full_state", ":", "Reference patch"),
        ]:
            ys = [float(next(r for r in rows if r["method"] == method and int(r["layer"]) == layer)[metric]) for layer in layers]
            ax.plot(layers, ys, label=label, linestyle=style, linewidth=2.0, color=colors[method])
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(layers)
    axes[0].set_ylabel("Match vs full")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Layer sensitivity of live handoff")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "layer_sweep_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "layer_sweep_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_suffix_span(rows: list[dict[str, str]]) -> None:
    sites = ["layer16", "layer34", "stride"]
    spans = [1, 4, 8]
    methods = ["identity", "mlp", "oracle"]
    metric = "mean_match_h8"

    fig, axes = plt.subplots(1, len(sites), figsize=(13, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    colors = {"identity": "#444444", "mlp": "#1f77b4", "oracle": "#d62728"}

    for ax, site in zip(axes, sites):
        x = range(len(spans))
        width = 0.24
        for idx, method in enumerate(methods):
            ys = []
            for span in spans:
                row = next(r for r in rows if r["site"] == site and int(r["span"]) == span and r["method"] == method)
                ys.append(float(row[metric]))
            ax.bar([v + (idx - 1) * width for v in x], ys, width=width, label=method if site == sites[0] else None, color=colors[method])
        ax.set_title(site)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(s) for s in spans])
        ax.set_xlabel("Suffix span")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Continuation match h8")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Suffix-span sensitivity at the selected intervention sites")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "suffix_span_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "suffix_span_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_regime_separation(rows: list[dict[str, str]]) -> None:
    layers = [int(r["layer"]) for r in rows]
    cosines = [float(r["residual_cosine_mean"]) for r in rows]
    mses = [float(r["residual_mse_mean"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)

    axes[0].plot(layers, cosines, marker="o", linewidth=2.0, color="#1f77b4")
    axes[0].set_title("Residual cosine by layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Cosine(low, full)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(layers, mses, marker="s", linewidth=2.0, color="#d62728")
    axes[1].set_title("Residual MSE by layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Residual MSE")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("RegimeLift low/full separation profile")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "regime_separation_profile.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "regime_separation_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def write_experiment_timeline() -> None:
    lines = [
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.18\textwidth}X>{\raggedright\arraybackslash}p{0.26\textwidth}}",
        r"\toprule",
        r"Phase & What was tested & Outcome \\",
        r"\midrule",
        r"0 & Low/full separation sanity check & Passed: the two regimes are meaningfully different. \\",
        r"1 & Extractor validation on structured prompts & Passed: paired residuals, caches, and logits line up. \\",
        r"2 & Null controls plus broad baseline table & Passed: identity equals no-patch; broad MLP is not enough. \\",
        r"3 & Layer sweep over candidate sites & Passed: layer 34 is the strongest learned site. \\",
        r"4 & Suffix-span sweep at the selected sites & Passed: last1 is already the informative span. \\",
        r"5 & Targeted study with held-out prompts & Passed: layer34,last1 generalizes; oracle is reference. \\",
        r"6 & Held-out80 rerun with fail-fast checks & Completed: 80 prompts loaded and evaluated. \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (GEN_DIR / "experiment_timeline.tex").write_text("\n".join(lines) + "\n")


def write_log_excerpt() -> None:
    lines = [
        r"\begin{verbatim}",
        r'resolved_heldout_extract_config = .../configs/extract_killtest40_holdout80.yaml',
        r'resolved_heldout_prompts_path = .../data/prompt_pool_v1_holdout80.jsonl',
        r'"heldout_prompt_count": 80',
        r'"phase": "heldout:complete"',
        r'"num_rows": 14',
        r'"heldout_prompt_id_first": "sym_0253"',
        r'"heldout_prompt_id_last": "con_0079"',
        r'[targeted] split=heldout prompts=80',
        r'Identity control passed',
        r'layer34,last1 is the useful learned site',
        r"\end{verbatim}",
    ]
    (GEN_DIR / "log_excerpt.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    targeted_rows = read_csv(TARGETED_ROWS)
    layer_rows = read_csv(LAYER_SWEEP_ROWS)
    suffix_rows = read_csv(SUFFIX_SPAN_ROWS)
    sanity_report = json.loads(SANITY_REPORT.read_text())
    sanity_layer_rows = read_csv(SANITY_LAYER_ROWS)

    write_targeted_core_table(targeted_rows)
    write_layer_sweep_summary_table(layer_rows)
    write_regime_separation_table(sanity_report)
    write_regime_controls_table(sanity_report)
    write_experiment_timeline()
    write_log_excerpt()

    plot_targeted_curves(targeted_rows)
    plot_targeted_delta_bars(targeted_rows)
    plot_layer_sweep(layer_rows)
    plot_suffix_span(suffix_rows)
    plot_regime_separation(sanity_layer_rows)

    print("Generated paper assets in", GEN_DIR)
    print("Generated paper figures in", FIG_DIR)


if __name__ == "__main__":
    main()
