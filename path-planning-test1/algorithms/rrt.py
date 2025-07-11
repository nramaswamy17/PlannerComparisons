import numpy as np
import time

def sample_free(grid):
    while True:
        x = np.random.randint(0, grid.shape[0])
        y = np.random.randint(0, grid.shape[1])
        if grid[x, y] == 0:
            return (x, y)

def nearest_vertex(tree, sample):
    return min(tree, key=lambda v: np.hypot(v[0]-sample[0], v[1]-sample[1]))

def steer(from_v, to_v, step=1):
    vec = np.array(to_v) - np.array(from_v)
    dist = np.linalg.norm(vec)
    if dist < step:
        return to_v
    else:
        vec = vec / dist * step
        return tuple(np.round(from_v + vec).astype(int))

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
    tree = {start: None}

    for _ in range(max_iters):
        sample = sample_free(grid)
        nearest = nearest_vertex(tree, sample)
        new_point = steer(nearest, sample)
        if collision_free(grid, nearest, new_point):
            tree[new_point] = nearest
            if np.hypot(new_point[0]-goal[0], new_point[1]-goal[1]) < 2 and collision_free(grid, new_point, goal):
                tree[goal] = new_point
                break

    path, node = [], goal
    if goal in tree:
        while node:
            path.append(node)
            node = tree[node]
        path.reverse()
        path_cost = sum(np.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) for i in range(1,len(path)))
    else:
        path = None
        path_cost = np.inf


    computation_time = time.time() - start_time
    return path, path_cost, computation_time
