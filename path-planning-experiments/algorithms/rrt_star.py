import numpy as np, time

# reuse helper functions from rrt.py (sample_free, nearest_vertex, steer, collision_free)

def near(tree, vertex, radius=5):
    return [v for v in tree if np.hypot(v[0]-vertex[0], v[1]-vertex[1]) <= radius]

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
