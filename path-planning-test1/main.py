from scenarios.generate_scenarios import generate_dynamic_scenario
from metrics.evaluation import evaluate_algorithm
from visualizations.plotting import plot_metric_comparison, plot_algorithms 
from algorithms import astar, theta_star, rrt, rrt_star, d_lite_star, lpa_star
from tqdm import tqdm

algorithm_funcs = {
    'A*': astar.plan,
    'Theta*': theta_star.plan#,
    #'RRT': rrt.plan,
    #'RRT*': rrt_star.plan,
    #'D-Lite*': d_lite_star.plan,
    #'LPA*': lpa_star.plan
}

SEED = 1  # Seed for reproducibility

scenarios = [generate_dynamic_scenario(obstacle_density=0, seed=SEED, size = (10,10), goal=(9,9)) for _ in range(10)]
start_goal_pairs = [((0, 0), (9, 9)) for _ in scenarios]

all_results = []
for name, func in tqdm(algorithm_funcs.items(), desc="Evaluating algorithms"):
    alg_results = evaluate_algorithm(func, scenarios, start_goal_pairs)
    all_results.append(alg_results)

#######
paths = []
grid = scenarios[0]
start, goal = start_goal_pairs[0]
# --- A* -------------------------------------------------------------
path_a, _, _ = astar.plan(grid, start, goal)          # returns [(r0,c0), (r1,c1), ...]
paths.append(("A*", path_a))
print("A* Complete")

# --- Theta* ---------------------------------------------------------
path_theta, _, _ = theta_star.plan(grid, start, goal)
paths.append(("Theta*", path_theta))
print("Theta* Complete")

# --- RRT ------------------------------------------------------------
#path_rrt, _, _ = rrt.plan(grid, start, goal)
#paths.append(("RRT", path_rrt))
print("RRT Complete")

# --- RRT* -----------------------------------------------------------
path_rrt_star, _, _ = rrt_star.plan(grid, start, goal)
paths.append(("RRT*", path_rrt_star))
print("RRT* Complete")

plot_algorithms(
    grid,
    start, goal, paths,
    figsize=(12, 8)
)
########


# Plot results for Path Length and Computation Time
#plot_metric_comparison(all_results, algorithm_funcs.keys(), metric_idx=0, metric_name='Path Length')
#plot_metric_comparison(all_results, algorithm_funcs.keys(), metric_idx=1, metric_name='Computation Time (s)')
