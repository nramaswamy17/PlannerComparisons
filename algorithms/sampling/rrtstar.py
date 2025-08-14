# algorithms/sampling_based/rrt_star.py
import numpy as np
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from core.base_planner import BasePlanner, PlanningResult
from core.environment import BaseEnvironment
from core.node import Node
from core.path import Path

class RRTStarNode(Node):
    """Extended Node class for RRT* with cost tracking and tree structure"""
    
    def __init__(self, position: np.ndarray, parent: Optional['RRTStarNode'] = None):
        super().__init__(position, parent)
        self.children: List['RRTStarNode'] = []
        self.cost: float = 0.0  # Cost from start to this node
        
    def add_child(self, child: 'RRTStarNode'):
        """Add a child node to this node"""
        self.children.append(child)
        child.parent = self
        
    def remove_child(self, child: 'RRTStarNode'):
        """Remove a child node from this node"""
        if child in self.children:
            self.children.remove(child)
            
    def update_cost_and_propagate(self, new_cost: float):
        """Update this node's cost and propagate to children"""
        cost_difference = new_cost - self.cost
        self.cost = new_cost
        
        # Propagate cost change to all children
        for child in self.children:
            child.update_cost_and_propagate(child.cost + cost_difference)


class RRTStar(BasePlanner):
    """
    RRT* (RRT Star) algorithm for optimal motion planning.
    
    RRT* extends RRT by gradually improving path quality through rewiring.
    It maintains the same probabilistic completeness as RRT but also provides
    asymptotic optimality - the solution converges to the optimal path as
    the number of samples approaches infinity.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RRT* planner.
        
        Config options:
            - max_iterations: Maximum number of tree expansion iterations (default: 5000)
            - step_size: Maximum distance to extend toward random sample (default: 1.0)
            - goal_tolerance: Distance tolerance for reaching goal (default: 1.0)
            - goal_bias: Probability of sampling goal directly (0.0-1.0) (default: 0.1)
            - gamma: Parameter for near neighbor radius calculation (default: 2.0)
            - rewire_radius_factor: Factor for rewiring radius (default: 1.5)
            - collision_check_resolution: Resolution for collision checking paths (default: 0.1)
            - dimension: Configuration space dimension (default: 2)
        """
        default_config = {
            'max_iterations': 5000,
            'step_size': 1.0,
            'goal_tolerance': 1.0,
            'goal_bias': 0.1,
            'gamma': 2.0,  # RRT* specific parameter
            'rewire_radius_factor': 1.5,  # Factor for rewiring radius
            'collision_check_resolution': 0.1,
            'dimension': 2  # Configuration space dimension
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # RRT* specific data structures
        self.tree_nodes: List[RRTStarNode] = []
        self.root_node: Optional[RRTStarNode] = None
        self.best_goal_node: Optional[RRTStarNode] = None
        self.best_path_cost: float = float('inf')
        
        # Statistics for analysis
        self.rewire_count: int = 0
        self.path_improvements: List[float] = []
        
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             environment: BaseEnvironment, **kwargs) -> PlanningResult:
        """
        Plan path using RRT* algorithm.
        
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
        self.root_node = RRTStarNode(start)
        self.root_node.cost = 0.0
        self.tree_nodes = [self.root_node]
        self.explored_nodes = []  # For visualization compatibility
        self.best_goal_node = None
        self.best_path_cost = float('inf')
        self.rewire_count = 0
        self.path_improvements = []
        
        # Main RRT* loop
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
                # Find near neighbors for rewiring
                near_radius = self._calculate_near_radius(len(self.tree_nodes))
                near_nodes = self._find_near_nodes(new_node.position, near_radius)
                
                # Choose best parent among near neighbors
                best_parent = self._choose_best_parent(new_node, near_nodes)
                if best_parent is not None:
                    new_node.parent = best_parent
                    new_node.cost = best_parent.cost + np.linalg.norm(new_node.position - best_parent.position)
                    best_parent.add_child(new_node)
                
                # Add to tree
                self.tree_nodes.append(new_node)
                self.explored_nodes.append(new_node)  # For visualization
                
                # Rewire tree to improve paths through new node
                self._rewire_tree(new_node, near_nodes)
                
                # Check if goal reached and update best path
                if self.is_goal_reached(new_node, self.config['goal_tolerance']):
                    if new_node.cost < self.best_path_cost:
                        self.best_goal_node = new_node
                        self.best_path_cost = new_node.cost
                        self.path_improvements.append(new_node.cost)
        
        # Return best path found
        if self.best_goal_node is not None:
            path = Path.from_node_chain(self.best_goal_node)
            return self.create_result(
                path=path,
                success=True,
                algorithm_name="RRT*",
                explored_nodes=self.explored_nodes.copy(),
                tree_nodes=self.tree_nodes.copy(),
                iterations=self.config['max_iterations'],
                final_tree_size=len(self.tree_nodes),
                best_cost=self.best_path_cost,
                rewire_count=self.rewire_count,
                path_improvements=self.path_improvements.copy()
            )
        
        # No path found
        return self.create_result(
            success=False,
            algorithm_name="RRT*",
            explored_nodes=self.explored_nodes.copy(),
            tree_nodes=self.tree_nodes.copy(),
            iterations=self.config['max_iterations'],
            final_tree_size=len(self.tree_nodes),
            rewire_count=self.rewire_count,
            termination_reason="no_path_found"
        )
    
    def _sample_random_point(self) -> np.ndarray:
        """Sample a random point in the configuration space"""
        x_bounds, y_bounds = self.environment.bounds
        
        x = random.uniform(x_bounds[0], x_bounds[1])
        y = random.uniform(y_bounds[0], y_bounds[1])
        
        return np.array([x, y])
    
    def _find_nearest_node(self, point: np.ndarray) -> RRTStarNode:
        """Find the nearest node in the tree to the given point"""
        nearest_node = self.tree_nodes[0]
        nearest_distance = np.linalg.norm(nearest_node.position - point)
        
        for node in self.tree_nodes[1:]:
            distance = np.linalg.norm(node.position - point)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _calculate_near_radius(self, num_nodes: int) -> float:
        """
        Calculate the radius for finding near neighbors.
        Uses the RRT* formula: min(gamma * sqrt(log(n)/n), step_size)
        """
        if num_nodes <= 1:
            return self.config['step_size']
        
        # RRT* optimal radius formula
        d = self.config['dimension']
        gamma = self.config['gamma']
        
        # Calculate unit ball volume in d dimensions
        unit_ball_volume = (math.pi ** (d/2)) / math.gamma(d/2 + 1)
        
        # Optimal gamma calculation (can be overridden by config)
        if gamma == 2.0:  # Default value, use optimal
            # Get environment volume (approximation)
            x_bounds, y_bounds = self.environment.bounds
            free_space_volume = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])
            gamma = 2 * ((1 + 1/d) * (free_space_volume / unit_ball_volume)) ** (1/d)
        
        radius = min(
            gamma * ((math.log(num_nodes) / num_nodes) ** (1/d)),
            self.config['step_size']
        )
        
        return radius
    
    def _find_near_nodes(self, point: np.ndarray, radius: float) -> List[RRTStarNode]:
        """Find all nodes within radius of the given point"""
        near_nodes = []
        
        for node in self.tree_nodes:
            distance = np.linalg.norm(node.position - point)
            if distance <= radius:
                near_nodes.append(node)
        
        return near_nodes
    
    def _choose_best_parent(self, new_node: RRTStarNode, 
                           near_nodes: List[RRTStarNode]) -> Optional[RRTStarNode]:
        """
        Choose the best parent for new_node from near_nodes based on cost.
        """
        best_parent = None
        best_cost = float('inf')
        
        for candidate_parent in near_nodes:
            # Calculate cost if new_node were connected to candidate_parent
            edge_cost = np.linalg.norm(new_node.position - candidate_parent.position)
            total_cost = candidate_parent.cost + edge_cost
            
            # Check if path is collision-free and improves cost
            if (total_cost < best_cost and 
                self._is_path_collision_free(candidate_parent.position, new_node.position)):
                best_parent = candidate_parent
                best_cost = total_cost
        
        return best_parent
    
    def _rewire_tree(self, new_node: RRTStarNode, near_nodes: List[RRTStarNode]):
        """
        Rewire the tree to improve paths through new_node.
        Check if any near nodes would have better cost if connected through new_node.
        """
        for near_node in near_nodes:
            if near_node == new_node or near_node == new_node.parent:
                continue
            
            # Calculate potential new cost for near_node through new_node
            edge_cost = np.linalg.norm(near_node.position - new_node.position)
            new_cost = new_node.cost + edge_cost
            
            # If cost improvement and collision-free path
            if (new_cost < near_node.cost and 
                self._is_path_collision_free(new_node.position, near_node.position)):
                
                # Remove near_node from its current parent
                if near_node.parent is not None:
                    near_node.parent.remove_child(near_node)
                
                # Connect near_node to new_node
                new_node.add_child(near_node)
                near_node.parent = new_node
                
                # Update costs propagating to children
                near_node.update_cost_and_propagate(new_cost)
                
                self.rewire_count += 1
    
    def _extend_tree(self, nearest_node: RRTStarNode, target_point: np.ndarray) -> Optional[RRTStarNode]:
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
        
        # Create new node with cost
        new_node = RRTStarNode(new_position, nearest_node)
        new_node.cost = nearest_node.cost + extend_distance
        
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
            'name': 'RRT*',
            'optimal': 'Asymptotically Optimal',
            'complete': 'Probabilistically Complete',
            'space_complexity': 'O(n)',
            'time_complexity': 'O(n log n)',
            'parameters': [
                'max_iterations', 'step_size', 'goal_tolerance', 
                'goal_bias', 'gamma', 'rewire_radius_factor', 'collision_check_resolution'
            ],
            'description': 'Optimal Rapidly-exploring Random Tree with rewiring',
            'year': 2010,
            'category': 'Sampling-Based',
            'advantages': [
                'Asymptotically optimal',
                'Probabilistically complete',
                'Gradually improves solution quality',
                'Works in continuous space'
            ],
            'disadvantages': [
                'Slower than basic RRT',
                'More complex implementation',
                'May require many iterations for good solutions'
            ]
        }


# Enhanced visualization for RRT*
class RRTStarVisualizer:
    """Specialized visualizer for RRT* tree structures and optimization"""
    
    @staticmethod
    def plot_rrt_star_result(result: PlanningResult, environment: BaseEnvironment,
                            start: np.ndarray, goal: np.ndarray, 
                            show_tree: bool = True, show_costs: bool = False):
        """Plot RRT* result with tree visualization and cost information"""
        
        import matplotlib.pyplot as plt
        from visualization.matplotlib_visualizer import MatplotlibVisualizer
        
        fig, ax = plt.subplots(figsize=(15, 10))
        viz = MatplotlibVisualizer(figsize=(15, 10))
        
        # Plot environment
        viz.plot_environment(environment, ax)
        
        # Plot tree if available
        if show_tree and 'tree_nodes' in result.metadata:
            tree_nodes = result.metadata['tree_nodes']
            
            # Plot tree edges with cost-based coloring
            if show_costs and tree_nodes:
                max_cost = max(node.cost for node in tree_nodes if hasattr(node, 'cost'))
                min_cost = min(node.cost for node in tree_nodes if hasattr(node, 'cost'))
                cost_range = max_cost - min_cost if max_cost > min_cost else 1
                
                for node in tree_nodes:
                    if node.parent is not None and hasattr(node, 'cost'):
                        # Color based on cost (blue = low cost, red = high cost)
                        normalized_cost = (node.cost - min_cost) / cost_range
                        color = plt.cm.coolwarm(normalized_cost)
                        
                        ax.plot([node.parent.position[0], node.position[0]],
                               [node.parent.position[1], node.position[1]],
                               color=color, alpha=0.6, linewidth=0.8, zorder=1)
            else:
                # Standard tree plotting
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
        
        # Plot final path
        if result.success and result.path:
            viz.plot_path(result.path, ax, color='red', linewidth=4,
                         label=f'RRT* Path (cost: {result.metadata.get("best_cost", 0):.2f})')
        
        # Plot start and goal
        ax.scatter(*start, c='blue', s=300, marker='o', 
                  edgecolors='white', linewidth=3, label='Start', zorder=5)
        ax.scatter(*goal, c='red', s=300, marker='*', 
                  edgecolors='white', linewidth=3, label='Goal', zorder=5)
        
        # Title and styling
        status = "SUCCESS" if result.success else "FAILED"
        tree_size = result.metadata.get('final_tree_size', 0)
        iterations = result.metadata.get('iterations', 0)
        rewires = result.metadata.get('rewire_count', 0)
        
        title = f"RRT* Algorithm - {status}\n"
        title += f"Tree Size: {tree_size} nodes, Iterations: {iterations}, "
        title += f"Rewires: {rewires}, Time: {result.computation_time:.3f}s"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_convergence_analysis(result: PlanningResult):
        """Plot RRT* convergence analysis showing path cost improvements"""
        
        import matplotlib.pyplot as plt
        
        if not result.success or 'path_improvements' not in result.metadata:
            print("No convergence data available")
            return None
        
        improvements = result.metadata['path_improvements']
        if not improvements:
            print("No path improvements recorded")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot cost improvements over time
        iterations = range(1, len(improvements) + 1)
        ax.plot(iterations, improvements, 'b-', linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Path Improvement Number', fontsize=12)
        ax.set_ylabel('Path Cost', fontsize=12)
        ax.set_title('RRT* Path Cost Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        if len(improvements) > 1:
            initial_cost = improvements[0]
            final_cost = improvements[-1]
            improvement_percent = ((initial_cost - final_cost) / initial_cost) * 100
            
            ax.annotate(f'Initial: {initial_cost:.2f}', 
                       xy=(1, initial_cost), xytext=(10, 10),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.annotate(f'Final: {final_cost:.2f}\n({improvement_percent:.1f}% improvement)', 
                       xy=(len(improvements), final_cost), xytext=(10, -20),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    from environments.simple_2d import Simple2D
    
    # Create environment and test scenario
    env = Simple2D.with_simple_obstacles(size=20)
    start, goal = env.create_test_scenario("challenging")
    
    # Create RRT* planner
    rrt_star = RRTStar({
        'max_iterations': 3000,
        'step_size': 1.5,
        'goal_bias': 0.1,
        'goal_tolerance': 1.5,
        'gamma': 2.0
    })
    
    # Plan path
    result = rrt_star.plan(start, goal, env)
    
    # Display results
    if result.success:
        print(f"✅ RRT* found path with cost: {result.metadata.get('best_cost', 0):.2f}")
        print(f"Tree size: {result.metadata.get('final_tree_size', 0)} nodes")
        print(f"Rewires performed: {result.metadata.get('rewire_count', 0)}")
        print(f"Path improvements: {len(result.metadata.get('path_improvements', []))}")
        print(f"Computation time: {result.computation_time:.3f}s")
    else:
        print("❌ RRT* failed to find path")
    
    # Visualize result
    fig = RRTStarVisualizer.plot_rrt_star_result(result, env, start, goal, show_tree=True)
    fig.show()