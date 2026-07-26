"""
Task 1: Investigate MDN Component Stability — Label Switching & Component Collapse

Parses per-epoch validation diagnostics from SLURM training logs and produces:
1. Component statistics tables across epochs
2. Label switching detection
3. Component collapse analysis (entropy, max weight)
4. Per-epoch component evolution plots

Usage:
  python investigate_mdn_stability.py \
    --residual_log results/slurm/mdn_residual_train_1074.out \
    --label_log results/slurm/mdn_attr_861.out \
    --output_dir results/stability_analysis
"""

import argparse
import re
import os
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def parse_epoch_blocks(log_path):
    """
    Parse a SLURM log file and extract per-epoch validation diagnostics.
    Returns list of dicts, one per epoch, containing per-attribute mixture stats.
    """
    with open(log_path, "r") as f:
        content = f.read()

    # Split on "=== Validation Diagnostics ==="
    blocks = content.split("=== Validation Diagnostics ===")
    if len(blocks) < 2:
        raise ValueError(f"No validation diagnostics found in {log_path}")

    epochs = []
    for block_idx, block in enumerate(blocks[1:], start=1):
        epoch_data = {"epoch": block_idx, "attributes": {}}

        # Extract target means and expected reward means
        m = re.search(r"Target Means \(Orig\):\s+\[([^\]]+)\]", block)
        if m:
            epoch_data["target_means"] = [float(x.strip()) for x in m.group(1).split(",")]

        m = re.search(r"Expected Reward Means \(Orig\):\s+\[([^\]]+)\]", block)
        if m:
            epoch_data["expected_means"] = [float(x.strip()) for x in m.group(1).split(",")]

        m = re.search(r"Mean Residuals \(Orig\):\s+\[([^\]]+)\]", block)
        if m:
            epoch_data["mean_residuals"] = [float(x.strip()) for x in m.group(1).split(",")]

        # Extract per-attribute pi, mu, s
        for attr in ATTRIBUTES:
            attr_data = {}

            # Find attribute section
            attr_pattern = rf"Attribute: {attr}\s*\n"
            attr_match = re.search(attr_pattern, block)
            if not attr_match:
                continue

            attr_block = block[attr_match.end():]
            # Limit to next attribute or end
            next_attr = re.search(r"  Attribute:", attr_block)
            if next_attr:
                attr_block = attr_block[:next_attr.start()]

            # pi mean
            m = re.search(r"pi mean:\s+\[([^\]]+)\]", attr_block)
            if m:
                attr_data["pi_mean"] = [float(x.strip()) for x in m.group(1).split(",")]

            # pi entropy
            m = re.search(r"pi entropy:\s+([\d.]+)", attr_block)
            if m:
                attr_data["pi_entropy"] = float(m.group(1))

            # mu mean
            m = re.search(r"mu mean:\s+\[([^\]]+)\]", attr_block)
            if m:
                attr_data["mu_mean"] = [float(x.strip()) for x in m.group(1).split(",")]

            # mu std
            m = re.search(r"mu std:\s+\[([^\]]+)\]", attr_block)
            if m:
                attr_data["mu_std"] = [float(x.strip()) for x in m.group(1).split(",")]

            # s/sigma mean
            m = re.search(r"s/sigma mean:\s+\[([^\]]+)\]", attr_block)
            if m:
                attr_data["s_mean"] = [float(x.strip()) for x in m.group(1).split(",")]

            # s/sigma std
            m = re.search(r"s/sigma std:\s+\[([^\]]+)\]", attr_block)
            if m:
                attr_data["s_std"] = [float(x.strip()) for x in m.group(1).split(",")]

            epoch_data["attributes"][attr] = attr_data

        epochs.append(epoch_data)

    return epochs


def detect_label_switching(epochs, attr):
    """
    Check if component identities swap between consecutive epochs.
    Uses component mean ordering to detect permutations.
    Returns list of (epoch_t, epoch_t+1, permutation) tuples.
    """
    switches = []
    for i in range(len(epochs) - 1):
        attr_t = epochs[i]["attributes"].get(attr, {})
        attr_t1 = epochs[i + 1]["attributes"].get(attr, {})

        if "mu_mean" not in attr_t or "mu_mean" not in attr_t1:
            continue

        mu_t = np.array(attr_t["mu_mean"])
        mu_t1 = np.array(attr_t1["mu_mean"])

        # Find best permutation by matching components (greedy nearest-mean)
        K = len(mu_t)
        order_t = np.argsort(mu_t)
        order_t1 = np.argsort(mu_t1)

        # If sorted orderings differ, components may have swapped
        if not np.array_equal(order_t, order_t1):
            switches.append({
                "epoch_from": epochs[i]["epoch"],
                "epoch_to": epochs[i + 1]["epoch"],
                "order_t": order_t.tolist(),
                "order_t1": order_t1.tolist(),
                "mu_t": mu_t.tolist(),
                "mu_t1": mu_t1.tolist(),
            })

    return switches


def analyze_collapse(epochs, attr):
    """
    Check for component collapse: max weight close to 1, entropy close to 0.
    """
    results = []
    for ep in epochs:
        attr_data = ep["attributes"].get(attr, {})
        if "pi_mean" not in attr_data:
            continue

        pi = np.array(attr_data["pi_mean"])
        max_pi = np.max(pi)
        entropy = attr_data.get("pi_entropy", None)
        max_possible_entropy = np.log(len(pi))

        results.append({
            "epoch": ep["epoch"],
            "max_pi": max_pi,
            "entropy": entropy,
            "max_entropy": max_possible_entropy,
            "normalized_entropy": entropy / max_possible_entropy if entropy else None,
            "collapsed": max_pi > 0.95,
        })

    return results


def plot_component_evolution(epochs, attr, output_dir, model_name):
    """
    Plot π, μ, s for each component across epochs for one attribute.
    """
    n_epochs = len(epochs)
    epoch_nums = [ep["epoch"] for ep in epochs]

    K = 3  # assume 3 components
    pi_series = [[] for _ in range(K)]
    mu_series = [[] for _ in range(K)]
    s_series = [[] for _ in range(K)]

    for ep in epochs:
        attr_data = ep["attributes"].get(attr, {})
        for k in range(K):
            pi_series[k].append(attr_data.get("pi_mean", [None]*K)[k])
            mu_series[k].append(attr_data.get("mu_mean", [None]*K)[k])
            s_series[k].append(attr_data.get("s_mean", [None]*K)[k])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#0f1117")
    colors = ["#64b5f6", "#a5d6a7", "#ef9a9a"]

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333344")
        ax.grid(color="#2a2d3a", linewidth=0.8, alpha=0.5)

    # π across epochs
    for k in range(K):
        axes[0].plot(epoch_nums, pi_series[k], marker="o", color=colors[k],
                     lw=2, label=f"Component {k+1}")
    axes[0].set_title("Mixture Weights (π)", color="white", fontsize=11)
    axes[0].set_ylabel("Average π", color="#cccccc")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(1/K, color="white", lw=0.8, linestyle=":", alpha=0.4, label="Uniform (1/K)")

    # μ across epochs
    for k in range(K):
        axes[1].plot(epoch_nums, mu_series[k], marker="s", color=colors[k],
                     lw=2, label=f"Component {k+1}")
    axes[1].set_title("Component Means (μ)", color="white", fontsize=11)
    axes[1].set_ylabel("Average μ", color="#cccccc")

    # s across epochs
    for k in range(K):
        axes[2].plot(epoch_nums, s_series[k], marker="^", color=colors[k],
                     lw=2, label=f"Component {k+1}")
    axes[2].set_title("Component Scales (s)", color="white", fontsize=11)
    axes[2].set_ylabel("Average s", color="#cccccc")

    for ax in axes:
        ax.set_xlabel("Epoch", color="#cccccc")
        ax.legend(fontsize=7.5, facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc")

    fig.suptitle(f"{model_name} — {attr}", color="white", fontsize=13, y=1.02)
    plt.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_{attr}_evolution.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return fname


def plot_collapse_summary(collapse_data_all, output_dir, model_name):
    """
    Plot entropy and max-weight histograms across all attributes and epochs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1117")
    colors_attr = ["#64b5f6", "#a5d6a7", "#ef9a9a", "#ffd54f", "#ce93d8"]

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333344")
        ax.grid(color="#2a2d3a", linewidth=0.8, alpha=0.5)

    for i, attr in enumerate(ATTRIBUTES):
        data = collapse_data_all[attr]
        epochs_list = [d["epoch"] for d in data]
        max_pis = [d["max_pi"] for d in data]
        entropies = [d["entropy"] for d in data if d["entropy"] is not None]
        entropy_epochs = [d["epoch"] for d in data if d["entropy"] is not None]

        axes[0].plot(epochs_list, max_pis, marker="o", color=colors_attr[i],
                     lw=2, label=attr)
        if entropies:
            axes[1].plot(entropy_epochs, entropies, marker="s", color=colors_attr[i],
                         lw=2, label=attr)

    axes[0].set_title("Max Component Weight (higher = more collapsed)", color="white", fontsize=10)
    axes[0].set_ylabel("max(π)", color="#cccccc")
    axes[0].set_xlabel("Epoch", color="#cccccc")
    axes[0].axhline(0.95, color="#ef5350", lw=1.5, linestyle="--", alpha=0.7, label="Collapse threshold (0.95)")
    axes[0].axhline(1/3, color="white", lw=0.8, linestyle=":", alpha=0.4, label="Uniform (1/3)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc", loc="lower right")

    axes[1].set_title("Mixture Entropy (lower = more collapsed)", color="white", fontsize=10)
    axes[1].set_ylabel("H(π)", color="#cccccc")
    axes[1].set_xlabel("Epoch", color="#cccccc")
    max_H = np.log(3)
    axes[1].axhline(max_H, color="#a5d6a7", lw=1, linestyle="--", alpha=0.5, label=f"Max entropy (ln3={max_H:.3f})")
    axes[1].axhline(0, color="#ef5350", lw=1, linestyle="--", alpha=0.5, label="Full collapse (H=0)")
    axes[1].legend(fontsize=7, facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc")

    fig.suptitle(f"{model_name} — Component Collapse Analysis", color="white", fontsize=13, y=1.02)
    plt.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_collapse_summary.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return fname


def generate_report(model_name, epochs, collapse_data, switching_data, plot_files, output_dir):
    """
    Generate a markdown report for one model.
    """
    lines = []
    lines.append(f"# {model_name}: Component Stability Analysis\n")

    # 1. Component stats table
    lines.append("## 1. Component Statistics Across Epochs\n")
    for attr in ATTRIBUTES:
        lines.append(f"### {attr.capitalize()}\n")
        lines.append("| Epoch | π₁ | π₂ | π₃ | μ₁ | μ₂ | μ₃ | s₁ | s₂ | s₃ | H(π) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ep in epochs:
            d = ep["attributes"].get(attr, {})
            pi = d.get("pi_mean", ["-"]*3)
            mu = d.get("mu_mean", ["-"]*3)
            s = d.get("s_mean", ["-"]*3)
            H = d.get("pi_entropy", "-")

            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else str(v)

            lines.append(f"| {ep['epoch']} | {fmt(pi[0])} | {fmt(pi[1])} | {fmt(pi[2])} | "
                         f"{fmt(mu[0])} | {fmt(mu[1])} | {fmt(mu[2])} | "
                         f"{fmt(s[0])} | {fmt(s[1])} | {fmt(s[2])} | {fmt(H)} |")
        lines.append("")

    # 2. Collapse analysis
    lines.append("## 2. Component Collapse Analysis\n")
    collapse_found = False
    for attr in ATTRIBUTES:
        data = collapse_data[attr]
        collapsed_epochs = [d for d in data if d.get("collapsed", False)]
        if collapsed_epochs:
            collapse_found = True
            lines.append(f"**{attr.capitalize()}**: ⚠️ Collapse detected at epochs "
                         f"{[d['epoch'] for d in collapsed_epochs]} "
                         f"(max π > 0.95)")
        else:
            max_max_pi = max(d["max_pi"] for d in data) if data else 0
            lines.append(f"**{attr.capitalize()}**: max(π) peaked at {max_max_pi:.4f} — "
                         f"{'⚠️ near collapse' if max_max_pi > 0.90 else '✅ no collapse'}")
    lines.append("")

    if not collapse_found:
        lines.append("> No full component collapse (max π > 0.95) detected for any attribute.\n")

    # 3. Label switching
    lines.append("## 3. Label Switching Analysis\n")
    any_switching = False
    for attr in ATTRIBUTES:
        switches = switching_data[attr]
        if switches:
            any_switching = True
            lines.append(f"**{attr.capitalize()}**: {len(switches)} switching event(s) detected:")
            for sw in switches:
                lines.append(f"  - Epoch {sw['epoch_from']}→{sw['epoch_to']}: "
                             f"mean order {sw['order_t']}→{sw['order_t1']}")
                lines.append(f"    μ values: {[f'{v:.2f}' for v in sw['mu_t']]} → "
                             f"{[f'{v:.2f}' for v in sw['mu_t1']]}")
        else:
            lines.append(f"**{attr.capitalize()}**: ✅ No label switching detected "
                         f"(component mean ordering stable)")
    lines.append("")

    if not any_switching:
        lines.append("> Component identities remain stable throughout training "
                     "(sorted component order unchanged between all epoch pairs).\n")

    # 4. Plot references
    lines.append("## 4. Evolution Plots\n")
    for pf in plot_files:
        lines.append(f"![{pf}]({os.path.join(output_dir, pf)})\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_log", type=str,
                        default="results/slurm/mdn_residual_train_1074.out")
    parser.add_argument("--label_log", type=str,
                        default="results/slurm/mdn_attr_861.out")
    parser.add_argument("--output_dir", type=str,
                        default="results/stability_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    full_report = ["# MDN Component Stability Investigation\n"]

    for model_name, log_path in [("Residual MDN", args.residual_log),
                                  ("Label MDN", args.label_log)]:
        if not os.path.exists(log_path):
            print(f"⚠️  Log not found: {log_path}, skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name} ({log_path})")
        print(f"{'='*60}")

        epochs = parse_epoch_blocks(log_path)
        print(f"  Found {len(epochs)} epoch(s)")

        # Analysis
        collapse_data = {}
        switching_data = {}
        plot_files = []

        for attr in ATTRIBUTES:
            # Collapse
            collapse_data[attr] = analyze_collapse(epochs, attr)

            # Label switching
            switching_data[attr] = detect_label_switching(epochs, attr)

            # Plot evolution
            fname = plot_component_evolution(epochs, attr, args.output_dir, model_name)
            plot_files.append(fname)
            print(f"  Generated: {fname}")

        # Collapse summary plot
        collapse_fname = plot_collapse_summary(collapse_data, args.output_dir, model_name)
        plot_files.append(collapse_fname)
        print(f"  Generated: {collapse_fname}")

        # Print summary to terminal
        print(f"\n--- {model_name} Summary ---")
        for attr in ATTRIBUTES:
            switches = switching_data[attr]
            collapses = [d for d in collapse_data[attr] if d.get("collapsed")]
            max_pi = max(d["max_pi"] for d in collapse_data[attr]) if collapse_data[attr] else 0
            min_entropy = min(d["entropy"] for d in collapse_data[attr]
                              if d["entropy"] is not None) if any(
                d["entropy"] is not None for d in collapse_data[attr]) else float("inf")

            status = []
            if collapses:
                status.append(f"COLLAPSED at epoch(s) {[c['epoch'] for c in collapses]}")
            if switches:
                status.append(f"{len(switches)} label switch(es)")
            if not status:
                status.append("STABLE")

            print(f"  {attr:15s}: max π={max_pi:.4f}, min H={min_entropy:.4f} — {', '.join(status)}")

        # Generate per-model report
        report = generate_report(model_name, epochs, collapse_data, switching_data,
                                 plot_files, args.output_dir)
        full_report.append(report)

        # Save component stats to CSV
        csv_path = os.path.join(args.output_dir, f"{model_name.lower().replace(' ', '_')}_stats.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Attribute", "Component",
                             "pi_mean", "mu_mean", "mu_std", "s_mean", "s_std",
                             "pi_entropy"])
            for ep in epochs:
                for attr in ATTRIBUTES:
                    d = ep["attributes"].get(attr, {})
                    K = len(d.get("pi_mean", []))
                    for k in range(K):
                        writer.writerow([
                            ep["epoch"], attr, k + 1,
                            d.get("pi_mean", [None]*K)[k],
                            d.get("mu_mean", [None]*K)[k],
                            d.get("mu_std", [None]*K)[k],
                            d.get("s_mean", [None]*K)[k],
                            d.get("s_std", [None]*K)[k],
                            d.get("pi_entropy", None),
                        ])
        print(f"  Saved: {csv_path}")

    # Write combined report
    report_path = os.path.join(args.output_dir, "stability_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(full_report))
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
