import numpy as np

def calculate_path_length(path):
    return sum(np.linalg.norm(np.subtract(path[i], path[i-1])) for i in range(1, len(path)))

def calculate_success_rate(paths):
    return sum(1 for path in paths if path is not None) / len(paths)

def evaluate_algorithm(algorithm, scenarios, start_goal_pairs):
    results = []
    for grid, (start, goal) in zip(scenarios, start_goal_pairs):
        try:
            path, cost, computation_time = algorithm(grid, start, goal)
            length = calculate_path_length(path) if path else np.inf
            results.append((length, computation_time, bool(path)))
        except:
            results.append((np.inf, np.inf, False))
    return results
