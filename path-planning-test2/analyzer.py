#!/usr/bin/env python3
# analyze_results.py
import argparse
import pathlib
import ast

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def analyze_results(csv_path: str):
    results_file = pathlib.Path(csv_path)
    if not results_file.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(results_file)
    df_success = df[df["success"] == True]

    # -----------------------------------------------------------------
    # Metrics summary (needed for the bar-charts on the first PDF page)
    # -----------------------------------------------------------------
    metrics = ["cost", "path_len", "expand_nodes", "time_sec"]
    avg_metrics = (
        df_success.groupby("planner")[metrics]
        .mean()
        .reset_index()
    )

    # -----------------------------------------------------------------
    # Build the PDF
    #   • Page 1  = 2 × 2 bar-chart collage (drawn in-memory)
    #   • Pages >1 = per-trial path visualisations
    # -----------------------------------------------------------------
    pdf_path = "planner_trial_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        # -------- Page 1: Bar-chart collage --------
        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        axs = axs.flatten()

        for i, metric in enumerate(metrics):
            sns.barplot(
                x="planner",
                y=metric,
                data=avg_metrics,
                ax=axs[i],
            )
            axs[i].set_title(f"Average {metric}", pad=6, fontsize=10)
            axs[i].set_xlabel("Planner")
            axs[i].set_ylabel(metric)

        # Hide any unused cells (in case metrics < 4)
        for j in range(len(metrics), len(axs)):
            axs[j].axis("off")

        fig.suptitle("Average Metrics by Planner", fontweight="bold", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # -------- Pages 2…N: per-trial path visualisations --------
        for trial in sorted(df_success["trial"].unique()):
            trial_data = df_success[df_success["trial"] == trial]
            planners = trial_data["planner"].unique()

            fig, axs = plt.subplots(
                1, len(planners),
                figsize=(5 * len(planners), 5)
            )
            if len(planners) == 1:
                axs = [axs]  # ensure iterable

            for ax, planner in zip(axs, planners):
                row = trial_data[trial_data["planner"] == planner].iloc[0]
                w, h = int(row["w"]), int(row["h"])

                ax.set_title(f"Trial {trial} – {planner}")
                ax.set_xlim(-1, w)
                ax.set_ylim(-1, h)
                ax.set_aspect("equal")
                ax.invert_yaxis()

                # Draw the path (if present)
                try:
                    path = (
                        ast.literal_eval(row["path"])
                        if isinstance(row["path"], str) else None
                    )
                    if path:
                        xs, ys = zip(*path)
                        ax.plot(xs, ys, "-", linewidth=2, color="blue", zorder=1)
                        ax.scatter(xs[0],  ys[0],  color="green", s=40,
                                   marker="s", zorder=10)  # start
                        ax.scatter(xs[-1], ys[-1], color="red",   s=40,
                                   marker="s", zorder=10)  # goal
                    else:
                        ax.text(0.5, 0.5, "(no path)",
                                ha="center", va="center", fontsize=10)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}",
                            ha="center", va="center", fontsize=10)

            pdf.savefig(fig)
            plt.close(fig)

    print(f"[✓] PDF written to {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to results.csv")
    args = parser.parse_args()

    analyze_results(args.csv_path)
