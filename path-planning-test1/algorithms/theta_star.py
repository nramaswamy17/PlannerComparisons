import heapq
import numpy as np
import time

def line_of_sight(grid, s, e):
    """Checks line of sight with Bresenham’s algorithm."""
    x0, y0 = s
    x1, y1 = e
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (-1 if x0 > x1 else 1), (-1 if y0 > y1 else 1)
    err = dx - dy

    while True:
        if grid[x0, y0]:
            return False
        if (x0, y0) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def neighbors(grid, x, y):
    nbrs = []
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    rows, cols = grid.shape
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
            nbrs.append((nx, ny))
    return nbrs

def plan(grid, start, goal):
    start_time = time.time()
    h = lambda p: np.hypot(p[0]-goal[0], p[1]-goal[1])
    openq = [(h(start), 0, start, start)]
    came, cost = {}, {start: 0}

    while openq:
        _, g, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if cur == goal:
            break
        for nxt in neighbors(grid, *cur):
            parent_of_cur = came[cur]
            if line_of_sight(grid, parent_of_cur, nxt):
                tentative_cost = cost[parent_of_cur] + np.hypot(nxt[0]-parent_of_cur[0], nxt[1]-parent_of_cur[1])
                par = parent_of_cur
            else:
                tentative_cost = cost[cur] + np.hypot(nxt[0]-cur[0], nxt[1]-cur[1])
                par = cur

            if nxt not in cost or tentative_cost < cost[nxt]:
                cost[nxt] = tentative_cost
                heapq.heappush(openq, (tentative_cost + h(nxt), tentative_cost, nxt, par))

    p, path = goal, []
    if goal in came:
        while p != start:
            path.append(p)
            p = came[p]
        path.append(start)
        path.reverse()
        path_cost = cost[goal]
    else:
        path = None
        path_cost = np.inf

    computation_time = time.time() - start_time
    return path, path_cost, computation_time
