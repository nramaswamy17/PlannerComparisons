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
    env = Simple2D.with_simple_obstacles(size=20)
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

