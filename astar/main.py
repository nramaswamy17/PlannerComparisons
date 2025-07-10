import argparse, time, pygame as pg
from grid import new_grid
import planners, metrics

TILE = 16
COLORS = {0: (40, 40, 40), 1: (200, 200, 200)}   # walkable / wall
PATH_COLOR  = (255, 140, 0)   # orange
START_COLOR = (0, 200, 60)    # green
GOAL_COLOR  = (200, 50, 0)    # red

START = (0, 0)                # goal is set per-run

def draw_grid(surf, g, path, start, goal):
    for y, row in enumerate(g):
        for x, val in enumerate(row):
            pg.draw.rect(surf, COLORS[val], (x*TILE, y*TILE, TILE, TILE))

    if path:
        for x, y in path:
            pg.draw.rect(surf, PATH_COLOR, (x*TILE, y*TILE, TILE, TILE))

    sx, sy = start
    gx, gy = goal
    pg.draw.rect(surf, START_COLOR, (sx*TILE, sy*TILE, TILE, TILE))
    pg.draw.rect(surf, GOAL_COLOR,  (gx*TILE, gy*TILE, TILE, TILE))

def run(algo, seed):
    grid  = new_grid(50, 35, seed=seed)
    start = START
    goal  = (grid.shape[1]-1, grid.shape[0]-1)

    t0   = time.time()
    path = getattr(planners, algo)(grid, start, goal)
    cpu_ms = (time.time() - t0) * 1000

    pg.init()
    w, h  = grid.shape[1]*TILE, grid.shape[0]*TILE
    screen = pg.display.set_mode((w, h))
    pg.display.set_caption(algo)
    clock = pg.time.Clock()

    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return len(path), cpu_ms
        screen.fill((20, 20, 20))
        draw_grid(screen, grid, path, start, goal)
        pg.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="a_star")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log = metrics.Logger("runs.csv")
    plen, t = run(args.algo, args.seed)
    log(args.algo, args.seed, plen, "N/A", f"{t:.1f}")
