# algorithms/sampling_based/rrt.py
import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
from core.base_planner import BasePlanner, PlanningResult
from core.environment import BaseEnvironment
from core.node import Node
from core.path import Path

class RRTNode(Node):
    """Extended Node class for RRT with tree structure tracking"""
    
    def __init__(self, position: np.ndarray, parent: Optional['RRTNode'] = None):
        super().__init__(position, parent)
        self.children: List['RRTNode'] = []
        
    def add_child(self, child: 'RRTNode'):
        """Add a child node to this node"""
        self.children.append(child)
        child.parent = self


class RRT(BasePlanner):
    """
    Rapidly-exploring Random Tree (RRT) algorithm for motion planning.
    
    RRT builds a tree by iteratively sampling random points in the configuration
    space and extending the tree toward these samples. This creates a probabilistically
    complete planning algorithm that works well in high-dimensional continuous spaces.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RRT planner.
        
        Config options:
            - max_iterations: Maximum number of tree expansion iterations (default: 5000)
            - step_size: Maximum distance to extend toward random sample (default: 1.0)
            - goal_tolerance: Distance tolerance for reaching goal (default: 1.0)
            - goal_bias: Probability of sampling goal directly (0.0-1.0) (default: 0.1)
            - max_extend_distance: Maximum distance for a single extension (default: 2.0)
            - collision_check_resolution: Resolution for collision checking paths (default: 0.1)
        """
        default_config = {
            'max_iterations': 5000,
            'step_size': 1.0,
            'goal_tolerance': 1.0,
            'goal_bias': 0.1,  # 10% chance to sample goal
            'max_extend_distance': 2.0,
            'collision_check_resolution': 0.1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # RRT-specific data structures
        self.tree_nodes: List[RRTNode] = []
        self.root_node: Optional[RRTNode] = None
        
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             environment: BaseEnvironment, **kwargs) -> PlanningResult:
        """
        Plan path using RRT algorithm.
        
        Args:
            start: Start position
            goal: Goal position  
            environment: Planning environment
            
        Returns:
            PlanningResult with path and metrics
        """
        # Setup
        self.setup_planning(start, goal, environment)
        
        # Initialize tree with start node
        self.root_node = RRTNode(start)
        self.tree_nodes = [self.root_node]
        self.explored_nodes = []  # For visualization compatibility
        
        # Main RRT loop
        for iteration in range(self.config['max_iterations']):
            self.current_iteration = iteration
            
            # Sample random point (with goal bias)
            if random.random() < self.config['goal_bias']:
                random_point = goal.copy()
            else:
                random_point = self._sample_random_point()
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(random_point)
            
            # Extend tree toward random point
            new_node = self._extend_tree(nearest_node, random_point)
            
            if new_node is not None:
                # Add to tree
                self.tree_nodes.append(new_node)
                nearest_node.add_child(new_node)
                self.explored_nodes.append(new_node)  # For visualization
                
                # Check if goal reached
                if self.is_goal_reached(new_node, self.config['goal_tolerance']):
                    # Found path to goal!
                    path = Path.from_node_chain(new_node)
                    return self.create_result(
                        path=path,
                        success=True,
                        algorithm_name="RRT",
                        explored_nodes=self.explored_nodes.copy(),
                        tree_nodes=self.tree_nodes.copy(),
                        iterations=iteration + 1,
                        final_tree_size=len(self.tree_nodes)
                    )
        
        # No path found within max iterations
        return self.create_result(
            success=False,
            algorithm_name="RRT",
            explored_nodes=self.explored_nodes.copy(),
            tree_nodes=self.tree_nodes.copy(),
            iterations=self.config['max_iterations'],
            final_tree_size=len(self.tree_nodes),
            termination_reason="max_iterations"
        )
    
    def _sample_random_point(self) -> np.ndarray:
        """Sample a random point in the configuration space"""
        x_bounds, y_bounds = self.environment.bounds
        
        x = random.uniform(x_bounds[0], x_bounds[1])
        y = random.uniform(y_bounds[0], y_bounds[1])
        
        return np.array([x, y])
    
    def _find_nearest_node(self, point: np.ndarray) -> RRTNode:
        """Find the nearest node in the tree to the given point"""
        nearest_node = self.tree_nodes[0]
        nearest_distance = np.linalg.norm(nearest_node.position - point)
        
        for node in self.tree_nodes[1:]:
            distance = np.linalg.norm(node.position - point)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _extend_tree(self, nearest_node: RRTNode, target_point: np.ndarray) -> Optional[RRTNode]:
        """
        Extend the tree from nearest_node toward target_point.
        
        Returns:
            New node if extension successful, None otherwise
        """
        # Calculate direction and distance
        direction = target_point - nearest_node.position
        distance = np.linalg.norm(direction)
        
        if distance == 0:
            return None
        
        # Normalize direction
        direction = direction / distance
        
        # Limit extension distance
        extend_distance = min(distance, self.config['step_size'])
        new_position = nearest_node.position + direction * extend_distance
        
        # Check if new position is within bounds
        if not self.environment.is_within_bounds(new_position):
            return None
        
        # Check if new position is collision-free
        if not self.environment.is_collision_free(new_position):
            return None
        
        # Check if path from nearest_node to new_position is collision-free
        if not self._is_path_collision_free(nearest_node.position, new_position):
            return None
        
        # Create new node
        new_node = RRTNode(new_position, nearest_node)
        return new_node
    
    def _is_path_collision_free(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        """
        Check if the straight-line path between two points is collision-free.
        Uses fine-grained collision checking.
        """
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        
        if distance == 0:
            return True
        
        # Number of collision checks based on resolution
        num_checks = max(2, int(distance / self.config['collision_check_resolution']))
        
        for i in range(num_checks + 1):
            alpha = i / num_checks
            check_point = start_pos + alpha * direction
            
            if not self.environment.is_collision_free(check_point):
                return False
        
        return True
    
    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all edges in the tree for visualization.
        
        Returns:
            List of (parent_pos, child_pos) tuples
        """
        edges = []
        for node in self.tree_nodes:
            if node.parent is not None:
                edges.append((node.parent.position, node.position))
        return edges
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        return {
            'name': 'RRT',
            'optimal': False,
            'complete': 'Probabilistically Complete',
            'space_complexity': 'O(n)',
            'time_complexity': 'O(n log n)',
            'parameters': [
                'max_iterations', 'step_size', 'goal_tolerance', 
                'goal_bias', 'max_extend_distance', 'collision_check_resolution'
            ],
            'description': 'Rapidly-exploring Random Tree for continuous space planning',
            'year': 1998,
            'category': 'Sampling-Based',
            'advantages': [
                'Works in continuous space',
                'Probabilistically complete', 
                'Good for high-dimensional spaces',
                'Fast for single queries'
            ],
            'disadvantages': [
                'Not optimal',
                'Path quality varies',
                'Sensitive to parameters'
            ]
        }


# Enhanced visualization for RRT
class RRTVisualizer:
    """Specialized visualizer for RRT tree structures"""
    
    @staticmethod
    def plot_rrt_result(result: PlanningResult, environment: BaseEnvironment,
                       start: np.ndarray, goal: np.ndarray, 
                       show_tree: bool = True, show_samples: bool = False):
        """Plot RRT result with tree visualization"""
        
        import matplotlib.pyplot as plt
        from visualization.matplotlib_visualizer import MatplotlibVisualizer
        
        viz = MatplotlibVisualizer(figsize=(14, 10))
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot environment
        viz.plot_environment(environment, ax)
        
        # Plot tree if available
        if show_tree and 'tree_nodes' in result.metadata:
            tree_nodes = result.metadata['tree_nodes']
            
            # Plot tree edges
            for node in tree_nodes:
                if node.parent is not None:
                    ax.plot([node.parent.position[0], node.position[0]],
                           [node.parent.position[1], node.position[1]],
                           'g-', alpha=0.3, linewidth=0.5, zorder=1)
            
            # Plot tree nodes
            if tree_nodes:
                tree_positions = np.array([node.position for node in tree_nodes])
                ax.scatter(tree_positions[:, 0], tree_positions[:, 1],
                          c='lightgreen', s=15, alpha=0.6, 
                          label=f'Tree Nodes ({len(tree_nodes)})', zorder=2)
        
        # Plot explored samples if requested
        if show_samples and result.metadata.get('explored_nodes'):
            explored = result.metadata['explored_nodes']
            if explored:
                explored_positions = np.array([node.position for node in explored])
                ax.scatter(explored_positions[:, 0], explored_positions[:, 1],
                          c='lightblue', s=8, alpha=0.4,
                          label=f'Samples ({len(explored)})', zorder=2)
        
        # Plot final path
        if result.success and result.path:
            viz.plot_path(result.path, ax, color='red', linewidth=4,
                         label=f'RRT Path (length: {result.path.length:.2f})')
        
        # Plot start and goal
        ax.scatter(*start, c='blue', s=300, marker='o', 
                  edgecolors='white', linewidth=3, label='Start', zorder=5)
        ax.scatter(*goal, c='red', s=300, marker='*', 
                  edgecolors='white', linewidth=3, label='Goal', zorder=5)
        
        # Title and styling
        status = "SUCCESS" if result.success else "FAILED"
        tree_size = result.metadata.get('final_tree_size', 0)
        iterations = result.metadata.get('iterations', 0)
        
        title = f"RRT Algorithm - {status}\n"
        title += f"Tree Size: {tree_size} nodes, Iterations: {iterations}, "
        title += f"Time: {result.computation_time:.3f}s"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Comprehensive test and comparison
if __name__ == "__main__":
    from environments.simple_2d import Simple2D
    from algorithms.classical.astar import AStar
    from algorithms.classical.dijkstra import Dijkstra
    import matplotlib.pyplot as plt
    
    def run_rrt_comprehensive_test():
        """Run comprehensive RRT testing and comparison"""
        
        print("🌳 RRT (Rapidly-exploring Random Tree) Testing")
        print("=" * 60)
        
        # Create test environment with obstacles
        env = Simple2D.with_simple_obstacles(size=20)
        start, goal = env.create_test_scenario("challenging")
        
        print(f"Environment: Simple2D with obstacles")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Direct distance: {np.linalg.norm(goal - start):.2f}")
        print()
        
        # Test different RRT configurations
        rrt_configs = [
            {
                'name': 'RRT Standard',
                'config': {
                    'max_iterations': 3000,
                    'step_size': 1.5,
                    'goal_bias': 0.1,
                    'goal_tolerance': 1.5
                }
            },
            {
                'name': 'RRT High Goal Bias',  
                'config': {
                    'max_iterations': 3000,
                    'step_size': 1.5,
                    'goal_bias': 0.3,  # Higher goal bias
                    'goal_tolerance': 1.5
                }
            },
            {
                'name': 'RRT Small Steps',
                'config': {
                    'max_iterations': 5000,
                    'step_size': 0.8,  # Smaller steps
                    'goal_bias': 0.1,
                    'goal_tolerance': 1.2
                }
            }
        ]
        
        # Also test classical algorithms for comparison
        classical_algorithms = [
            (AStar({'step_size': 1.0, 'goal_tolerance': 1.5, 'max_iterations': 10000}), "A*"),
            (Dijkstra({'step_size': 1.0, 'goal_tolerance': 1.5, 'max_iterations': 10000}), "Dijkstra")
        ]
        
        all_results = []
        
        # Test RRT variants
        for config_info in rrt_configs:
            print(f"🌲 Testing {config_info['name']}...")
            
            rrt = RRT(config_info['config'])
            result = rrt.plan(start, goal, env)
            all_results.append((result, config_info['name']))
            
            if result.success:
                print(f"  ✅ SUCCESS - Path length: {result.path.length:.2f}")
                print(f"     Tree size: {result.metadata.get('final_tree_size', 0)} nodes")
                print(f"     Iterations: {result.metadata.get('iterations', 0)}")
                print(f"     Time: {result.computation_time:.3f}s")
            else:
                print(f"  ❌ FAILED - Tree size: {result.metadata.get('final_tree_size', 0)}")
            print()
        
        # Test classical algorithms for comparison
        print("🔗 Testing Classical Algorithms for Comparison...")
        for planner, name in classical_algorithms:
            print(f"  Testing {name}...")
            result = planner.plan(start, goal, env)
            all_results.append((result, name))
            
            if result.success:
                print(f"    ✅ SUCCESS - Path length: {result.path.length:.2f}")
                print(f"       Nodes explored: {result.nodes_explored}")
                print(f"       Time: {result.computation_time:.3f}s")
            else:
                print(f"    ❌ FAILED")
            print()
        
        # Create visualizations
        print("📊 Creating Visualizations...")
        
        # Individual RRT visualizations
        for result, name in all_results[:len(rrt_configs)]:  # Only RRT results
            if 'RRT' in name:
                fig = RRTVisualizer.plot_rrt_result(result, env, start, goal, 
                                                   show_tree=True, show_samples=False)
                filename = f"rrt_{name.lower().replace(' ', '_')}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  Saved {filename}")
        
        # Algorithm comparison
        successful_results = [(r, n) for r, n in all_results if r.success]
        if len(successful_results) > 1:
            from visualization.matplotlib_visualizer import MatplotlibVisualizer
            viz = MatplotlibVisualizer()
            fig = viz.plot_multiple_paths(
                [(r.path, n) for r, n in successful_results],
                env, start, goal, 
                title="RRT vs Classical Algorithms Comparison"
            )
            fig.savefig("rrt_vs_classical_comparison.png", dpi=300, bbox_inches='tight')
            print("  Saved rrt_vs_classical_comparison.png")
        
        # Performance summary
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"{'Algorithm':<20} {'Success':<8} {'Path Length':<12} {'Nodes/Tree':<12} {'Time (s)':<10}")
        print("-" * 70)
        
        for result, name in all_results:
            success = "✅" if result.success else "❌"
            path_len = f"{result.path.length:.2f}" if result.success else "N/A"
            
            if 'RRT' in name:
                nodes_info = f"{result.metadata.get('final_tree_size', 0)}"
            else:
                nodes_info = f"{result.nodes_explored}"
            
            time_taken = f"{result.computation_time:.3f}"
            
            print(f"{name:<20} {success:<8} {path_len:<12} {nodes_info:<12} {time_taken:<10}")
        
        print("\n🎯 KEY INSIGHTS:")
        rrt_results = [(r, n) for r, n in all_results if 'RRT' in n and r.success]
        classical_results = [(r, n) for r, n in all_results if 'RRT' not in n and r.success]
        
        if rrt_results and classical_results:
            avg_rrt_time = np.mean([r.computation_time for r, n in rrt_results])
            avg_classical_time = np.mean([r.computation_time for r, n in classical_results])
            
            print(f"  ⚡ Average RRT time: {avg_rrt_time:.3f}s")
            print(f"  🔗 Average Classical time: {avg_classical_time:.3f}s")
            
            if avg_rrt_time < avg_classical_time:
                speedup = avg_classical_time / avg_rrt_time
                print(f"  🚀 RRT is {speedup:.1f}x faster on average!")
            
            # Path quality comparison
            rrt_paths = [r.path.length for r, n in rrt_results]
            classical_paths = [r.path.length for r, n in classical_results]
            
            if rrt_paths and classical_paths:
                print(f"  📏 RRT path lengths: {min(rrt_paths):.2f} - {max(rrt_paths):.2f}")
                print(f"  📐 Classical path lengths: {min(classical_paths):.2f} - {max(classical_paths):.2f}")
        
        return all_results
    
    # Run the comprehensive test
    results = run_rrt_comprehensive_test()
    
    # Show plots
    plt.show()