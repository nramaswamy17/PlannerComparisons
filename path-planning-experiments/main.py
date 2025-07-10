from scenarios.generate_scenarios import generate_dynamic_scenario
from metrics.evaluation import evaluate_algorithm
from visualizations.plotting import plot_metric_comparison
from algorithms import astar, theta_star, rrt, rrt_star, d_lite_star, lpa_star
from tqdm import tqdm

algorithm_funcs = {
    'A*': astar.plan,
    'Theta*': theta_star.plan,
    'RRT': rrt.plan,
    'RRT*': rrt_star.plan,
    #'D-Lite*': d_lite_star.plan,
    #'LPA*': lpa_star.plan
}

SEED = 1  # Seed for reproducibility

scenarios = [generate_dynamic_scenario(seed=SEED) for _ in range(10)]
start_goal_pairs = [((0, 0), (49, 49)) for _ in scenarios]

all_results = []
for name, func in tqdm(algorithm_funcs.items(), desc="Evaluating algorithms"):
    alg_results = evaluate_algorithm(func, scenarios, start_goal_pairs)
    all_results.append(alg_results)

print(alg_results)

# Plot results for Path Length and Computation Time
plot_metric_comparison(all_results, algorithm_funcs.keys(), metric_idx=0, metric_name='Path Length')
plot_metric_comparison(all_results, algorithm_funcs.keys(), metric_idx=1, metric_name='Computation Time (s)')
