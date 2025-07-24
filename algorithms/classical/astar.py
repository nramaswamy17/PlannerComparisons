# algorithms/classical/astar.py
import heapq
import numpy as np
from typing import Dict, Any, List, Set
from core.base_planner import BasePlanner, PlanningResult
from core.environment import BaseEnvironment
from core.node import Node
from core.path import Path

class AStar(BasePlanner):
    """
    A* algorithm implementation for motion planning.
    Uses heuristic to guide search toward goal, maintaining optimality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize A* planner.
        
        Config options:
            - step_size: Step size for expanding nodes (default: 1.0)
            - max_iterations: Maximum iterations before giving up (default: 10000)
            - goal_tolerance: Distance tolerance for reaching goal (default: 0.5)
            - heuristic: Heuristic function ('euclidean', 'manhattan', 'octile') (default: 'euclidean')
            - heuristic_weight: Weight for heuristic (1.0 = optimal, >1.0 = faster but suboptimal)
            - allow_diagonal: Allow diagonal movement (default: True)
        """
        default_config = {
            'step_size': 1.0,
            'max_iterations': 10000,
            'goal_tolerance': 0.5,
            'heuristic': 'euclidean',
            'heuristic_weight': 1.0,
            'allow_diagonal': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.open_set: List[Node] = []
        self.closed_set: Set[str] = set()
        self.cost_so_far: Dict[str, float] = {}
        self.nodes_in_open: Dict[str, Node] = {}  # Track nodes in open set for updates
    
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             environment: BaseEnvironment, **kwargs) -> PlanningResult:
        """
        Plan path using A* algorithm.
        
        Args:
            start: Start position
            goal: Goal position  
            environment: Planning environment
            
        Returns:
            PlanningResult with path and metrics
        """
        # Setup
        self.setup_planning(start, goal, environment)
        
        # Initialize data structures
        self.open_set = []
        self.closed_set = set()
        self.cost_so_far = {}
        self.nodes_in_open = {}
        self.explored_nodes = []
        
        # Add start node with heuristic
        self.start_node.cost = 0.0
        self.start_node.heuristic = self._calculate_heuristic(self.start_node)
        
        heapq.heappush(self.open_set, (self.start_node.f_score, self.start_node))
        self.cost_so_far[self.start_node.id] = 0.0
        self.nodes_in_open[self.start_node.id] = self.start_node
        
        # Main planning loop
        while self.open_set and self.current_iteration < self.config['max_iterations']:
            self.current_iteration += 1
            
            # Get node with lowest f-score
            current_f_score, current_node = heapq.heappop(self.open_set)
            
            # Remove from open set tracking
            if current_node.id in self.nodes_in_open:
                del self.nodes_in_open[current_node.id]
            
            # Skip if already processed or if this is an outdated entry
            if current_node.id in self.closed_set:
                continue
            
            # Add to closed set
            self.closed_set.add(current_node.id)
            self.explored_nodes.append(current_node)
            
            # Check if goal reached
            if self.is_goal_reached(current_node, self.config['goal_tolerance']):
                path = Path.from_node_chain(current_node)
                return self.create_result(
                    path=path, 
                    success=True,
                    algorithm_name="A*",
                    explored_nodes=self.explored_nodes.copy(),
                    final_cost=current_node.cost,
                    final_f_score=current_node.f_score,
                    heuristic_type=self.config['heuristic'],
                    heuristic_weight=self.config['heuristic_weight']
                )
            
            # Expand neighbors
            neighbors = self._get_neighbors(current_node)
            
            for neighbor in neighbors:
                # Skip if in closed set
                if neighbor.id in self.closed_set:
                    continue
                
                # Skip if not collision free
                if not self.environment.is_collision_free(neighbor):
                    continue
                
                # Calculate costs
                edge_cost = current_node.distance_to(neighbor)
                tentative_g_cost = current_node.cost + edge_cost
                
                # Check if this path to neighbor is better
                if (neighbor.id not in self.cost_so_far or 
                    tentative_g_cost < self.cost_so_far[neighbor.id]):
                    
                    # Update neighbor
                    neighbor.parent = current_node
                    neighbor.cost = tentative_g_cost
                    neighbor.heuristic = self._calculate_heuristic(neighbor)
                    self.cost_so_far[neighbor.id] = tentative_g_cost
                    
                    # Add to open set (or update if already there)
                    if neighbor.id in self.nodes_in_open:
                        # Node is already in open set, need to update priority
                        # Python's heapq doesn't support decrease-key, so we'll add duplicate
                        # The outdated entry will be skipped when popped
                        pass
                    
                    heapq.heappush(self.open_set, (neighbor.f_score, neighbor))
                    self.nodes_in_open[neighbor.id] = neighbor
        
        # No path found
        return self.create_result(
            success=False,
            algorithm_name="A*",
            explored_nodes=self.explored_nodes.copy(),
            heuristic_type=self.config['heuristic'],
            heuristic_weight=self.config['heuristic_weight'],
            termination_reason="max_iterations" if self.current_iteration >= self.config['max_iterations'] else "no_path"
        )
    
    def _calculate_heuristic(self, node: Node) -> float:
        """Calculate heuristic value for a node"""
        return (self.config['heuristic_weight'] * 
                self.get_heuristic(node, self.goal_node, self.config['heuristic']))
    
    def _get_neighbors(self, node: Node) -> List[Node]:
        """
        Get neighboring nodes for expansion.
        Uses 8-connected grid if diagonal movement allowed, otherwise 4-connected.
        """
        neighbors = []
        step_size = self.config['step_size']
        
        # 4-connected (cardinal directions)
        directions = [
            (step_size, 0),      # East
            (-step_size, 0),     # West  
            (0, step_size),      # North
            (0, -step_size),     # South
        ]
        
        # Add diagonal directions if allowed
        if self.config['allow_diagonal']:
            diagonals = [
                (step_size, step_size),      # Northeast
                (-step_size, step_size),     # Northwest
                (step_size, -step_size),     # Southeast
                (-step_size, -step_size)     # Southwest
            ]
            directions.extend(diagonals)
        
        for dx, dy in directions:
            new_position = node.position + np.array([dx, dy])
            
            # Check bounds
            if not self.environment.is_within_bounds(new_position):
                continue
            
            # Create neighbor node
            neighbor = Node(new_position)
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        return {
            'name': 'A*',
            'optimal': self.config['heuristic_weight'] <= 1.0,
            'complete': True,
            'space_complexity': 'O(b^d)',
            'time_complexity': 'O(b^d)',
            'parameters': [
                'step_size', 'max_iterations', 'goal_tolerance', 
                'heuristic', 'heuristic_weight', 'allow_diagonal'
            ],
            'description': 'Best-first search using heuristic to guide exploration',
            'year': 1968,
            'category': 'Classical Graph-Based',
            'advantages': ['Optimal (with admissible heuristic)', 'Complete', 'Efficient'],
            'disadvantages': ['Memory intensive', 'Requires good heuristic', 'Discrete space only']
        }


# algorithms/classical/weighted_astar.py
class WeightedAStar(AStar):
    """
    Weighted A* - trades optimality for speed by weighting heuristic more heavily.
    Popular in robotics for real-time applications.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Weighted A* with higher default heuristic weight"""
        default_config = {
            'heuristic_weight': 1.5,  # Default to weighted version
            'step_size': 1.5,
            'max_iterations': 10000,
            'goal_tolerance': 0.5,
            'heuristic': 'euclidean',
            'allow_diagonal': True
        }
        
        if config:
            default_config.update(config)
        
        # Initialize parent AStar with weighted config
        super().__init__(default_config)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information for Weighted A*"""
        info = super().get_algorithm_info()
        info.update({
            'name': 'Weighted A*',
            'optimal': False,  # Not optimal due to weighting
            'description': 'A* with weighted heuristic for faster but suboptimal solutions',
            'advantages': ['Faster than A*', 'Good solution quality', 'Tunable speed/quality tradeoff'],
            'disadvantages': ['Not optimal', 'Solution quality depends on weight', 'Requires tuning']
        })
        return info


# Example usage and test script for A*
if __name__ == "__main__":
    # Test the complete system for A*
    from environments.simple_2d import Simple2D
    from visualization.matplotlib_visualizer import MatplotlibVisualizer

    # Create environment
    env = Simple2D.empty_environment(size=20)
    start, goal = env.create_test_scenario("simple")

    # Create planner
    astar = AStar({
        'step_size': 1.0,
        'goal_tolerance': 1.5,
        'max_iterations': 100000,
        'heuristic': 'euclidean'
    })

    # Plan path
    print("Planning path with A*...")
    result = astar.plan(start, goal, env)

    # Print results
    print(f"Planning result: {result}")
    if result.success:
        print(f"Path found! Length: {result.path.length:.2f}")
        print(f"Nodes explored: {result.nodes_explored}")
        print(f"Computation time: {result.computation_time:.3f} seconds")
    else:
        print("No path found!")

    # Visualize
    viz = MatplotlibVisualizer()
    fig = viz.plot_planning_result(result, env, start, goal,
                                   title="A* Algorithm Demo")

    # Show plot
    import matplotlib.pyplot as plt
    plt.show()

    # Save figure
    fig.savefig('astar_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'astar_demo.png'")

"""
# Example usage and comprehensive comparison
if __name__ == "__main__":
    import time
    from environments.simple_2d import Simple2D
    from visualization.matplotlib_visualizer import MatplotlibVisualizer
    from algorithms.classical.dijkstra import Dijkstra
    
    def run_algorithm_comparison():
        #"Run comprehensive comparison of classical algorithms"
        print("🚀 Motion Planning Algorithm Comparison")
        print("=" * 50)
        
        # Create test environment
        env = Simple2D.empty_environment(size=20)
        start, goal = env.create_test_scenario("challenging")
        
        print(f"Environment: Simple2D with obstacles")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Distance: {np.linalg.norm(goal - start):.2f}")
        print()
        
        # Algorithm configurations
        algorithms = [
            (Dijkstra({'step_size': 0.8, 'goal_tolerance': 1.0}), "Dijkstra"),
            (AStar({'step_size': 0.8, 'goal_tolerance': 1.0, 'heuristic': 'euclidean'}), "A* (Euclidean)"),
            (AStar({'step_size': 0.8, 'goal_tolerance': 1.0, 'heuristic': 'manhattan'}), "A* (Manhattan)"),
            (AStar({'step_size': 0.8, 'goal_tolerance': 1.0, 'heuristic': 'octile'}), "A* (Octile)"),
            (WeightedAStar({'step_size': 0.8, 'goal_tolerance': 1.0, 'heuristic_weight': 1.5}), "Weighted A* (1.5x)"),
            (WeightedAStar({'step_size': 0.8, 'goal_tolerance': 1.0, 'heuristic_weight': 2.0}), "Weighted A* (2.0x)")
        ]
        
        results = []
        
        # Run each algorithm
        for planner, name in algorithms:
            print(f"Running {name}...")
            start_time = time.time()
            result = planner.plan(start, goal, env)
            end_time = time.time()
            
            results.append((result, name))
            
            # Print results
            if result.success:
                print(f"  ✅ SUCCESS - Path length: {result.path.length:.2f}")
                print(f"     Nodes explored: {result.nodes_explored}")
                print(f"     Time: {result.computation_time:.3f}s")
                print(f"     Planning rate: {result.planning_rate:.0f} nodes/sec")
            else:
                print(f"  ❌ FAILED - {result.metadata.get('termination_reason', 'unknown')}")
            print()
        
        # Create visualizations
        viz = MatplotlibVisualizer(figsize=(15, 10))
        
        # Individual algorithm plots
        for result, name in results:
            if result.success:
                fig = viz.plot_planning_result(result, env, start, goal, 
                                             title=f"{name} - Path Planning Result")
                filename = f"{name.lower().replace(' ', '_').replace('*', 'star').replace('(', '').replace(')', '')}_result.png"
                viz.save_figure(fig, filename)
                print(f"Saved {filename}")
        
        # Comprehensive comparison
        successful_results = [(r, n) for r, n in results if r.success]
        if len(successful_results) > 1:
            fig = viz.plot_algorithm_comparison(successful_results, env, start, goal)
            viz.save_figure(fig, "algorithm_comparison.png")
            print("Saved algorithm_comparison.png")
            
            # Show comparison
            import matplotlib.pyplot as plt
            plt.show()
        
        # Print summary table
        print("\n📊 ALGORITHM COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Algorithm':<20} {'Success':<8} {'Path Length':<12} {'Nodes':<8} {'Time (s)':<10} {'Rate (n/s)':<12}")
        print("-" * 80)
        
        for result, name in results:
            success = "✅" if result.success else "❌"
            path_len = f"{result.path.length:.2f}" if result.success else "N/A"
            nodes = result.nodes_explored
            time_taken = f"{result.computation_time:.3f}"
            rate = f"{result.planning_rate:.0f}" if result.success else "N/A"
            
            print(f"{name:<20} {success:<8} {path_len:<12} {nodes:<8} {time_taken:<10} {rate:<12}")
        
        print("\n🎯 KEY INSIGHTS:")
        successful = [r for r, n in results if r.success]
        if successful:
            fastest = min(successful, key=lambda x: x.computation_time)
            shortest = min(successful, key=lambda x: x.path.length)
            most_efficient = min(successful, key=lambda x: x.nodes_explored)
            
            fastest_name = next(n for r, n in results if r == fastest)
            shortest_name = next(n for r, n in results if r == shortest)
            efficient_name = next(n for r, n in results if r == most_efficient)
            
            print(f"  🏃 Fastest: {fastest_name} ({fastest.computation_time:.3f}s)")
            print(f"  📏 Shortest path: {shortest_name} ({shortest.path.length:.2f})")
            print(f"  🧠 Most efficient: {efficient_name} ({most_efficient.nodes_explored} nodes)")
            
            # Calculate heuristic efficiency
            dijkstra_result = next((r for r, n in results if "Dijkstra" in n and r.success), None)
            if dijkstra_result:
                print(f"\n  💡 A* vs Dijkstra node exploration reduction:")
                for result, name in results:
                    if result.success and "A*" in name and "Weighted" not in name:
                        reduction = (1 - result.nodes_explored / dijkstra_result.nodes_explored) * 100
                        print(f"     {name}: {reduction:.1f}% fewer nodes explored")
    
    def run_heuristic_analysis():
        #Analyze different heuristic functions
        print("\n🧭 HEURISTIC FUNCTION ANALYSIS")
        print("=" * 50)
        
        # Create simple corridor environment to show heuristic effects
        env = Simple2D.corridor_environment()
        start, goal = env.create_test_scenario("corridor")
        
        heuristics = ['euclidean', 'manhattan', 'octile']
        
        for heuristic in heuristics:
            planner = AStar({
                'step_size': 0.5,
                'heuristic': heuristic,
                'goal_tolerance': 0.8
            })
            
            result = planner.plan(start, goal, env)
            
            print(f"{heuristic.capitalize()} heuristic:")
            if result.success:
                print(f"  Path length: {result.path.length:.2f}")
                print(f"  Nodes explored: {result.nodes_explored}")
                print(f"  Time: {result.computation_time:.3f}s")
            else:
                print("  Failed to find path")
            print()
    
    # Run the comprehensive analysis
    run_algorithm_comparison()
    run_heuristic_analysis()
"""