# Summary

This aims to cover what I think the key history of path planning algorithms is from Dijkstra's Algorithm until present day to prove I'm not clueless... :P

# Algorithms
## Classical Graph Based Algorithms (1950-1989)
- Dijkstra's Algorithm (1956)
- A* (1968)
- D*

## Sampling-Based Motion Planning (1990 - 2009)
- Probabilistic Roadmap (PRM)
- Rapidly-exploring Random Trees (RRT)
- RRT*
- RRT-Connect
- Informed RRT

## Optimization-Based Approaches (2000-2019)
- CHOMP - Covariant Hamiltonian Optimization for Motion Planning
- STOMP - Stochastic Trajectory Optimization for Motion Planning
- TrajOpt - Sequential convex optimization for trajectory planning

## Modern Hybrid and Learning-Based Methods (2010s-Present)
- FMT* - Fast Marching Trees
- BIT* - Batch Informed Trees
- Neural RRT*
- MPNet - Motion Planning Networks (NNs)
- Lightning Framework

## State of the Art (What I think is cutting edge now)
- MPPI - Model Predictive Path Integral
- Diffusion Policy - Diffusion for Trajectory Generation
- DDP / iLQG - Differential Dynamic Programming
- CBF-QP - Control Barrier Functions with QP (safety guarantees)



# Directory Structure
```
motion_planning_portfolio/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── algorithm_configs/
│   │   ├── __init__.py
│   │   ├── dijkstra_config.py
│   │   ├── astar_config.py
│   │   ├── rrt_config.py
│   │   └── ... (one per algorithm)
│   └── environment_configs/
│       ├── __init__.py
│       ├── simple_2d_config.py
│       ├── obstacle_course_config.py
│       └── warehouse_config.py
├── core/
│   ├── __init__.py
│   ├── base_planner.py          # Abstract base class
│   ├── environment.py           # Environment representation
│   ├── node.py                 # Node/state representation
│   ├── path.py                 # Path representation
│   ├── metrics.py              # Performance metrics
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py
│       ├── collision_detection.py
│       ├── data_structures.py
│       └── math_utils.py
├── algorithms/
│   ├── __init__.py
│   ├── classical/              # 1950-1989
│   │   ├── __init__.py
│   │   ├── dijkstra.py
│   │   ├── astar.py
│   │   └── dstar.py
│   ├── sampling_based/         # 1990-2009
│   │   ├── __init__.py
│   │   ├── prm.py
│   │   ├── rrt.py
│   │   ├── rrt_star.py
│   │   ├── rrt_connect.py
│   │   └── informed_rrt.py
│   ├── optimization_based/     # 2000-2019
│   │   ├── __init__.py
│   │   ├── chomp.py
│   │   ├── stomp.py
│   │   └── trajopt.py
│   └── modern_hybrid/          # 2010s-Present
│       ├── __init__.py
│       ├── fmt_star.py
│       ├── bit_star.py
│       ├── neural_rrt.py
│       ├── mpnet.py
│       └── lightning.py
├── environments/
│   ├── __init__.py
│   ├── base_environment.py
│   ├── simple_2d.py
│   ├── obstacle_course_2d.py
│   ├── warehouse_2d.py
│   └── maps/                   # Static map files
│       ├── simple_map.json
│       ├── warehouse.json
│       └── obstacle_course.json
├── visualization/
│   ├── __init__.py
│   ├── base_visualizer.py
│   ├── matplotlib_visualizer.py
│   ├── interactive_visualizer.py
│   └── animation_utils.py
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_runner.py
│   ├── performance_metrics.py
│   ├── test_scenarios/
│   │   ├── __init__.py
│   │   ├── simple_scenarios.py
│   │   ├── complex_scenarios.py
│   │   └── waymo_like_scenarios.py
│   └── results/                # Benchmark outputs
│       ├── performance_tables/
│       ├── plots/
│       └── animations/
├── examples/
│   ├── __init__.py
│   ├── quick_start.py
│   ├── algorithm_comparison.py
│   ├── custom_environment_demo.py
│   ├── benchmark_all_algorithms.py
│   └── interactive_demo.py
├── tests/
│   ├── __init__.py
│   ├── test_algorithms/
│   │   ├── test_dijkstra.py
│   │   ├── test_astar.py
│   │   └── ... (one per algorithm)
│   ├── test_environments/
│   ├── test_visualization/
│   └── integration_tests/
├── docs/
│   ├── README.md
│   ├── algorithm_explanations/
│   │   ├── classical_methods.md
│   │   ├── sampling_based.md
│   │   ├── optimization_based.md
│   │   └── modern_methods.md
│   ├── api_reference/
│   └── tutorials/
└── scripts/
    ├── run_single_algorithm.py
    ├── run_comparison.py
    ├── generate_benchmark_report.py
    └── create_demo_video.py    
```


## Implementation 

Getting Started Implementation Order

Core Infrastructure: base_planner.py, environment.py, node.py, path.py
Simple Environment: simple_2d.py - just start/goal points in free space
Basic Visualization: matplotlib_visualizer.py
First Algorithm: Dijkstra (simplest to implement correctly)
Add Obstacles: Extend environment with basic collision detection
Second Algorithm: A* (builds on Dijkstra)
Benchmarking Framework: Compare Dijkstra vs A*
Continue with RRT, RRT, etc.*


