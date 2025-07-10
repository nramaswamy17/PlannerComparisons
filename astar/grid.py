import numpy as np, random, json

def new_grid(w, h, p_block=0.15, seed=None):
    rng = np.random.default_rng(seed)
    g = np.zeros((h, w), dtype=np.uint8)
    mask = rng.random((h, w)) < p_block
    g[mask] = 1          # 1 = obstacle
    g[0, 0] = g[-1, -1] = 0   # keep start/goal free
    return g


def load_json(path):
    return np.array(json.load(open(path)), dtype=np.uint8)