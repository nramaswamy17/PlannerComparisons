from __future__ import annotations
import argparse, itertools, random, time, pathlib
from typing import Tuple, List, Dict, Type

import numpy as np
import pandas as pd
import python_motion_planning as pmp  # pip install python-motion-planning

###############################################################################
#  Planner registry – only discrete planners that are stable
###############################################################################
GRAPH_PLANNERS: Dict[str, Type[pmp.BasePlanner]] = {
    "AStar": pmp.AStar,
    "ThetaStar": pmp.ThetaStar,
}

###############################################################################
#  Helper functions
###############################################################################

def make_static_grid(w: int, h: int, density: float, rng: random.Random) -> pmp.Grid:
    env       = pmp.Grid(w, h)
    num_obs   = int(density * w * h)
    obstacles = {(rng.randint(0, w - 1), rng.randint(0, h - 1)) for _ in range(num_obs)}
    env.update(obstacles)
    env._w, env._h = w, h  # type: ignore[attr-defined]
    return env


def clone_grid(src: pmp.Grid) -> pmp.Grid:
    tgt = pmp.Grid(src._w, src._h)  # type: ignore[attr-defined]
    obstacles = set(src.obstacles)
    tgt.update(obstacles)
    tgt._w, tgt._h = src._w, src._h  # type: ignore[attr-defined]
    return tgt


def pick_free_coord(env: pmp.Grid, rng: random.Random) -> Tuple[int, int]:
    """Pick a random free cell."""
    while True:
        x, y = rng.randrange(env._w), rng.randrange(env._h)  # type: ignore[attr-defined]
        if (x, y) not in env.obstacles:
            return (x, y)

###############################################################################
#  Core execution
###############################################################################

def run_once(
    PlannerCls: Type[pmp.BasePlanner],
    env: pmp.Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
):
    """Run one planner, capture basic metrics **and the computed path**."""
    planner = PlannerCls(start=start, goal=goal, env=env)
    tic = time.perf_counter()
    result = planner.plan()
    toc = time.perf_counter()

    # Unpack result variants -------------------------------------------------
    if isinstance(result, tuple):
        if isinstance(result[0], (int, float)):
            cost, path, expanded = result
        else:
            path, cost, expanded = result
    else:  # Planner returned object / None on failure
        cost, path, expanded = float("inf"), [], []

    # Ensure path is a plain list of tuples so it can round-trip via CSV
    path_serialisable: List[Tuple[int, int]] = []
    if isinstance(path, (list, tuple)):
        try:
            path_serialisable = [(int(x), int(y)) for x, y in path]
        except Exception:
            path_serialisable = []

    return {
        "success": cost < float("inf"),
        "cost": cost,
        "path_len": len(path_serialisable) if path_serialisable else np.inf,
        "expand_nodes": len(expanded) if hasattr(expanded, "__len__") else expanded,
        "time_sec": toc - tic,
        "path": path_serialisable,  # <‑‑ added
        "error": "",
    }

###############################################################################
#  Main experiment loop
###############################################################################

def main(args):
    rng  = random.Random(42)
    rows = []

    for trial in range(args.trials):
        for w, h, dens in itertools.product(args.grid_sizes, args.grid_sizes, args.densities):
            base_env = make_static_grid(w, h, dens, rng)
            start    = pick_free_coord(base_env, rng)
            goal     = pick_free_coord(base_env, rng)

            for name, Planner in GRAPH_PLANNERS.items():
                try:
                    row = {
                        "trial": trial,
                        "planner": name,
                        "w": w,
                        "h": h,
                        "density": dens,
                    }
                    row.update(run_once(Planner, clone_grid(base_env), start, goal))
                    rows.append(row)
                except Exception as exc:
                    import traceback
                    print("=" * 70)
                    print(f"[!] Error during planner: {name} on {w}×{h} grid, density={dens}, trial={trial}")
                    traceback.print_exc()
                    print("=" * 70)
                    rows.append({
                        "trial": trial,
                        "planner": name,
                        "w": w,
                        "h": h,
                        "density": dens,
                        "success": False,
                        "error": str(exc),
                    })

    df = pd.DataFrame(rows)
    out = pathlib.Path("results.csv")
    df.to_csv(out, index=False)
    print(f"[✓] Saved raw results to {out.resolve()}")

###############################################################################
#  Entrypoint
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--densities", nargs="+", type=float, default=[0.2])
    main(parser.parse_args())
