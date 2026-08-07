import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import ks_2samp  # noqa: E402

from tabular_polygraph.dataset import load_dataset  # noqa: E402

# Set style for scientific paper
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def plot_empirical_gap(save_path):
    # Set global seeds for perfect reproducibility of the scientific figure
    np.random.seed(42)
    import random

    random.seed(42)

    print("Loading Real Data...")
    # Load supermarket sales dataset for the perfect visual spread
    real_df = load_dataset("supermarket_sales").dropna()

    # We will plot quantity vs total
    col_x = "quantity"
    col_y = "total"

    print("Training Generator (Empirical Gaussian Copula)...")
    from scipy import stats

    # 1. Transform real data to uniform using empirical CDF (ranks)
    u_x = stats.rankdata(real_df[col_x]) / (len(real_df) + 1.0)
    u_y = stats.rankdata(real_df[col_y]) / (len(real_df) + 1.0)

    # 2. Transform to Gaussian latent space
    n_x = stats.norm.ppf(u_x)
    n_y = stats.norm.ppf(u_y)

    # 3. Learn linear covariance matrix
    cov = np.cov(n_x, n_y)

    print("Generating Synthetic Data...")
    # 4. Sample from latent space and inverse transform via empirical quantiles
    syn_n = np.random.multivariate_normal([0, 0], cov, size=len(real_df))
    syn_u_x = np.clip(stats.norm.cdf(syn_n[:, 0]), 1e-6, 1 - 1e-6)
    syn_u_y = np.clip(stats.norm.cdf(syn_n[:, 1]), 1e-6, 1 - 1e-6)

    syn_df = pd.DataFrame()
    syn_df[col_x] = np.quantile(real_df[col_x], syn_u_x).round().astype(int)
    syn_df[col_y] = np.quantile(real_df[col_y], syn_u_y)

    # Calculate KS scores to prove marginal fidelity is high
    ks_x = 1.0 - ks_2samp(real_df[col_x], syn_df[col_x]).statistic
    ks_y = 1.0 - ks_2samp(real_df[col_y], syn_df[col_y]).statistic
    print(f"KS {col_x}: {ks_x:.3f}")
    print(f"KS {col_y}: {ks_y:.3f}")

    # For visualization, do NOT use jitter. The strict vertical alignment of integer quantities
    # emphasizes the arithmetic nature of the constraint.
    x_real = real_df[col_x]
    y_real = real_df[col_y]

    x_syn = syn_df[col_x]
    y_syn = syn_df[col_y]

    # Create the figure
    fig = plt.figure(figsize=(12, 5))
    # width_ratios: [Joint_Real(4), Marg_Y_Real(1), Spacer(0.5), Joint_Syn(4), Marg_Y_Syn(1)]
    gs = GridSpec(
        2,
        5,
        width_ratios=[4, 1, 0.5, 4, 1],
        height_ratios=[0.8, 4],
        wspace=0.1,
        hspace=0.1,
    )

    # --- REAL DATA PLOT ---
    ax_joint_real = fig.add_subplot(gs[1, 0])
    ax_marg_x_real = fig.add_subplot(gs[0, 0], sharex=ax_joint_real)
    ax_marg_y_real = fig.add_subplot(gs[1, 1], sharey=ax_joint_real)

    sns.scatterplot(
        x=x_real,
        y=y_real,
        ax=ax_joint_real,
        color="#1f77b4",
        alpha=0.15,
        edgecolor="none",
        s=25,
    )
    ax_joint_real.set_xlabel("Quantity (units)")
    ax_joint_real.set_ylabel("Total ($)")
    ax_joint_real.set_xlim(0, 11)
    ax_joint_real.set_ylim(-50, 1200)
    ax_joint_real.set_xticks(range(1, 11, 2))

    # Discrete histogram for quantity, KDE for subtotal
    sns.histplot(
        x=x_real,
        ax=ax_marg_x_real,
        discrete=True,
        color="#1f77b4",
        alpha=0.6,
        element="bars",
    )
    sns.kdeplot(
        y=y_real,
        ax=ax_marg_y_real,
        fill=True,
        color="#1f77b4",
        alpha=0.6,
        linewidth=1.5,
    )

    ax_marg_x_real.axis("off")
    ax_marg_y_real.axis("off")
    ax_marg_x_real.set_title(
        "Real Supermarket Data\n(Strict Arithmetic Dependency)",
        pad=15,
        fontweight="bold",
    )

    # --- SYNTHETIC DATA PLOT ---
    ax_joint_syn = fig.add_subplot(gs[1, 3], sharex=ax_joint_real, sharey=ax_joint_real)
    ax_marg_x_syn = fig.add_subplot(gs[0, 3], sharex=ax_joint_syn)
    ax_marg_y_syn = fig.add_subplot(gs[1, 4], sharey=ax_joint_syn)

    sns.scatterplot(
        x=x_syn,
        y=y_syn,
        ax=ax_joint_syn,
        color="#d62728",
        alpha=0.15,
        edgecolor="none",
        s=25,
    )
    ax_joint_syn.set_xlabel("Quantity (units)")
    ax_joint_syn.set_ylabel("Total ($)")

    sns.histplot(
        x=x_syn,
        ax=ax_marg_x_syn,
        discrete=True,
        color="#d62728",
        alpha=0.6,
        element="bars",
    )
    sns.kdeplot(
        y=y_syn,
        ax=ax_marg_y_syn,
        fill=True,
        color="#d62728",
        alpha=0.6,
        linewidth=1.5,
        clip=(0, None),
    )

    ax_marg_x_syn.axis("off")
    ax_marg_y_syn.axis("off")
    ax_marg_x_syn.set_title(
        f"Synthetic Data (Gaussian Copula)\n(KS: Qty {ks_x:.2f}, Total {ks_y:.2f}, but Broken Dependency)",
        pad=15,
        fontweight="bold",
    )

    # Annotate impossible region (Low Quantity, High Total)
    # Real max for Q=1 is ~100, Q=2 is ~200. Anything > 250 is impossible here.
    ax_joint_syn.add_patch(
        plt.Rectangle(
            (0.5, 250), 2.0, 750, fill=False, edgecolor="black", linestyle="--", lw=2
        )
    )

    # Add a custom text box pointing to the region
    ax_joint_syn.annotate(
        "Dependency Violation:\nImpossible Total",
        xy=(1.5, 400),
        xytext=(4.0, 700),
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "black", "lw": 1},
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3",
            "color": "black",
            "lw": 1.5,
        },
    )

    # Fix layout spacing
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15)
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved publication-quality figure to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the Dependency Gap figure")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(PROJECT_ROOT) / "manuscript" / "figures" / "dependency_gap.pdf",
        help="Output PDF path (default: manuscript/figures/dependency_gap.pdf)",
    )
    args = parser.parse_args()

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_empirical_gap(out_path)
