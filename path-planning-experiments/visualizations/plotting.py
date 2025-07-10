import matplotlib.pyplot as plt
import numpy as np
import math

def plot_metric_comparison(results, algorithms, metric_idx, metric_name):
    means, std_devs = [], []
    for alg_results in results:
        valid_results = [res[metric_idx] for res in alg_results if res[metric_idx] != np.inf]
        if valid_results:
            means.append(np.mean(valid_results))
            std_devs.append(np.std(valid_results))
        else:
            means.append(0)
            std_devs.append(0)

    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, means, yerr=std_devs, capsize=5)
    plt.ylabel(metric_name)
    plt.title(f'Algorithm Comparison by {metric_name}')
    plt.grid(axis='y', linestyle='--')
    plt.show()


# --- simple palette ----------------------------------------------------------
C_FREE      = (0.15, 0.15, 0.15)
C_OBSTACLE  = (1.00, 1.00, 1.00)
C_PATH      = (0.00, 0.55, 1.00)
C_START     = (0.00, 0.80, 0.00)
C_GOAL      = (0.90, 0.10, 0.10)

def plot_algorithms(grid, start, goal, algo_paths, figsize=(10, 10)):
    """
    Visualise *any* number of motion-planning solutions on the same occupancy grid.

    Parameters
    ----------
    grid        : 2-D (H×W) binary array, 0 = free, 1 = obstacle
    start, goal : (row, col) tuples
    algo_paths  : iterable of (name, path) where
                  - name : str  — subplot title (e.g. "A*")
                  - path : list/array of (row, col) points
    figsize     : overall Figure size in inches
    """
    algo_paths = list(algo_paths)  # in case caller passes a generator
    n          = len(algo_paths)
    if n == 0:
        raise ValueError("algo_paths is empty — nothing to plot!")

    # ---- choose a near-square layout ----------------------------------------
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=figsize,
                            squeeze=False)
    axs = axs.ravel()  # 1-D iterator through all slots

    # Pre-render the occupancy grid once as an RGB image
    rgb           = np.zeros(grid.shape + (3,), dtype=float)
    rgb[grid == 0] = C_FREE
    rgb[grid == 1] = C_OBSTACLE

    for ax, (name, path) in zip(axs, algo_paths):
        ax.imshow(rgb, origin="lower")
        if path:                              # skip empty paths
            ys, xs = zip(*path)
            ax.plot(xs, ys, lw=1.8, color=C_PATH)

        ax.scatter(start[1], start[0], c=[C_START], marker='o', s=40, zorder=3)
        ax.scatter(goal[1],  goal[0],  c=[C_GOAL],  marker='x', s=60, zorder=3)

        ax.set_title(name)
        ax.set_xticks([]);  ax.set_yticks([])

    # Turn off any unused panels (nrows*ncols may exceed n)
    for ax in axs[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()