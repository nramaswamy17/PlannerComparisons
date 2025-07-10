import heapq
import numpy as np
import time

def neighbors(grid, x, y):
    nbrs = []
    directions = [(1,0), (-1,0), (0,1), (0,-1)]  # 4-connected grid
    rows, cols = grid.shape
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
            nbrs.append((nx, ny))
    return nbrs

def plan(grid, start, goal):
    start_time = time.time()

    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    openq = [(h(start), 0, start, None)]
    came, cost = {}, {start: 0}

    while openq:
        _, g, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if cur == goal:
            break
        for nxt in neighbors(grid, *cur):
            new_cost = g + 1  # uniform cost grid; adjust if needed
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                heapq.heappush(openq, (new_cost + h(nxt), new_cost, nxt, cur))

    # reconstruct path
    p, path = goal, []
    if goal in came:
        while p:
            path.append(p)
            p = came.get(p)
        path = path[::-1]
        path_cost = len(path) - 1
    else:
        path = None
        path_cost = np.inf  # unreachable

    computation_time = time.time() - start_time

    return path, path_cost, computation_time
