import numpy as np, time
import random
import math

# reuse helper functions from rrt.py (sample_free, nearest_vertex, steer, collision_free)

def near(tree, vertex, radius=5):
    return [v for v in tree if np.hypot(v[0]-vertex[0], v[1]-vertex[1]) <= radius]

def nearest_vertex(tree, sample):
    return min(tree, key=lambda v: np.hypot(v[0]-sample[0], v[1]-sample[1]))

def steer(from_pt, to_pt, step_size=5):
    """
    Return a new grid cell that lies at most `step_size`
    cells away from `from_pt`, in the direction of `to_pt`.

    Parameters
    ----------
    from_pt   : (row, col)  – start of the edge
    to_pt     : (row, col)  – random sample
    step_size : int         – max distance to advance (≥ 1)

    Returns
    -------
    (row, col)  – the candidate new vertex
    """
    # Already there? (can happen when sample == from_pt)
    if from_pt == to_pt:
        return from_pt

    dr = to_pt[0] - from_pt[0]
    dc = to_pt[1] - from_pt[1]
    dist = math.hypot(dr, dc)

    # Scale the vector down to the desired step length.
    scale = min(1.0, step_size / dist)
    new_r = int(round(from_pt[0] + dr * scale))
    new_c = int(round(from_pt[1] + dc * scale))
    return (new_r, new_c)

def sample_free(grid):
    """
    Return a random (row, col) that is free (grid value == 0).

    Parameters
    ----------
    grid : 2-D ndarray of 0/1 ints

    Returns
    -------
    (int, int)  — coordinates of a free cell
    """
    rows, cols = grid.shape
    while True:
        r = random.randrange(rows)
        c = random.randrange(cols)
        if grid[r, c] == 0:
            return (r, c)

def collision_free(grid, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy  = x2 - x1, y2 - y1
    steps   = max(abs(dx), abs(dy))

    # Same cell?  Just check that single spot.
    if steps == 0:
        return grid[x1, y1] == 0      # True if free

    for i in range(steps + 1):
        t = i / steps
        x = int(round(x1 + t * dx))
        y = int(round(y1 + t * dy))
        if grid[x, y] == 1:
            return False
    return True

def plan(grid, start, goal, max_iters=2000):
    start_time = time.time()
    tree, cost = {start: None}, {start: 0}

    for _ in range(max_iters):
        sample = sample_free(grid)
        nearest = nearest_vertex(tree, sample)
        new_point = steer(nearest, sample)
        if not collision_free(grid, nearest, new_point):
            continue

        min_cost, min_parent = cost[nearest] + np.hypot(new_point[0]-nearest[0], new_point[1]-nearest[1]), nearest

        for v in near(tree, new_point):
            if collision_free(grid, v, new_point):
                new_cost = cost[v] + np.hypot(new_point[0]-v[0], new_point[1]-v[1])
                if new_cost < min_cost:
                    min_cost, min_parent = new_cost, v

        tree[new_point], cost[new_point] = min_parent, min_cost

        for v in near(tree, new_point):
            potential_cost = cost[new_point] + np.hypot(new_point[0]-v[0], new_point[1]-v[1])
            if potential_cost < cost.get(v, np.inf) and collision_free(grid, new_point, v):
                tree[v], cost[v] = new_point, potential_cost

        if np.hypot(new_point[0]-goal[0], new_point[1]-goal[1]) < 2 and collision_free(grid, new_point, goal):
            tree[goal], cost[goal] = new_point, cost[new_point] + np.hypot(new_point[0]-goal[0], new_point[1]-goal[1])
            break

    path, node = [], goal
    if goal in tree:
        while node:
            path.append(node)
            node = tree[node]
        path.reverse()
        path_cost = cost[goal]
    else:
        path, path_cost = None, np.inf

    computation_time = time.time() - start_time
    return path, path_cost, computation_time
