import heapq, math, numpy as np

# --- shared helpers ---------------------------------------------------------
dirs = [(1,0),(-1,0),(0,1),(0,-1)]  # 4-connected
def in_bounds(g, x, y): return 0 <= x < g.shape[1] and 0 <= y < g.shape[0]
def neighbors(g, x, y):
    for dx,dy in dirs:
        nx, ny = x+dx, y+dy
        if in_bounds(g, nx, ny) and g[ny,nx] == 0:
            yield (nx, ny)

# --- A* planner -------------------------------------------------------------
def a_star(grid, start, goal):
    h = lambda p: abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    openq = [(h(start), 0, start, None)]
    came, cost = {}, {start: 0}
    while openq:
        _, g, cur, parent = heapq.heappop(openq)
        if cur in came:      # already expanded
            continue
        came[cur] = parent
        if cur == goal:
            break
        for nxt in neighbors(grid, *cur):
            new_cost = g + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                heapq.heappush(openq, (new_cost + h(nxt), new_cost, nxt, cur))
    # reconstruct
    p, path = goal, []
    while p: path.append(p); p = came.get(p)
    return path[::-1]

# --- Dijkstra falls out “for free” -----------------------------------------
def dijkstra(grid, start, goal):
    return a_star(grid, start, goal)  # h = 0 implicit above
