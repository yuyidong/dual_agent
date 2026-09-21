from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "phase3_matrix_actual_full"


def main():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    summary = json.loads((ROOT / "phase3_cross_evaluation_summary.json").read_text())
    matrix = np.load(ROOT / "phase3_cross_evaluation_matrix.npz")["normalized_operating_cost"]
    mmd = summary["forecaster_output_mmd_from_F0"]
    size = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(7.0, 6.3), dpi=220)
    fig.subplots_adjust(left=0.15, right=0.84, bottom=0.14, top=0.87)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=float(matrix.min()), vmax=float(matrix.max()))

    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels([f"$F_{{{i}}}$\n$D={mmd[i]:.2f}$" for i in range(size)], color="#2166ac")
    ax.set_yticklabels([f"$S_{{{i}}}$" for i in range(size)])
    ax.set_xlabel(r"Forecaster state ($D$: MMD from $F_0$)", labelpad=11)
    ax.set_ylabel("Surrogate state", labelpad=10)
    ax.tick_params(length=0, pad=5)
    ax.set_title("Cross-evaluation of iterative adaptation", fontsize=14, fontweight="bold", pad=16)

    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    midpoint = (float(matrix.min()) + float(matrix.max())) / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if matrix[i, j] >= midpoint else "#102a43"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10, color=color)

    # Diagonal cells are the matched states; arrows show surrogate adaptation at fixed F_i.
    for i in range(size):
        ax.add_patch(
            plt.Rectangle((i - 0.42, i - 0.42), 0.84, 0.84, fill=False,
                          edgecolor="#202020", linewidth=1.1, zorder=4)
        )
    for i in range(1, size):
        ax.annotate(
            "", xy=(i, i - 0.26), xytext=(i, i - 1 + 0.26),
            arrowprops={"arrowstyle": "->", "color": "#202020", "lw": 1.2},
            zorder=5,
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.05)
    colorbar.set_label("Normalized dispatch cost", labelpad=10)
    colorbar.outline.set_linewidth(0.7)

    output = ROOT / "phase3_cross_evaluation_matrix_paper.png"
    output_pdf = ROOT / "phase3_cross_evaluation_matrix_paper.pdf"
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)
    print(output_pdf)


if __name__ == "__main__":
    main()
