
# algorithms/classical/dijkstra.py
import heapq
import numpy as np
from typing import Dict, Any, List, Set
from core.base_planner import BasePlanner, PlanningResult
from core.environment import BaseEnvironment
from core.node import Node
from core.path import Path

class Dijkstra(BasePlanner):
    """
    Dijkstra's algorithm implementation for motion planning.
    Guarantees shortest path in discrete space.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Dijkstra planner.
        
        Config options:
            - step_size: Step size for expanding nodes (default: 1.0)
            - max_iterations: Maximum iterations before giving up (default: 10000)
            - goal_tolerance: Distance tolerance for reaching goal (default: 0.5)
            - connection_radius: Maximum distance to connect nodes (default: 2.0)
        """
        default_config = {
            'step_size': 1.0,
            'max_iterations': 10000,
            'goal_tolerance': 0.5,
            'connection_radius': 2.0
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.open_set: List[Node] = []
        self.closed_set: Set[str] = set()
        self.cost_so_far: Dict[str, float] = {}
    
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             environment: BaseEnvironment, **kwargs) -> PlanningResult:
        """
        Plan path using Dijkstra's algorithm.
        
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
        self.explored_nodes = []
        
        # Add start node
        self.start_node.cost = 0.0
        heapq.heappush(self.open_set, (0.0, self.start_node))
        self.cost_so_far[self.start_node.id] = 0.0
        
        # Main planning loop
        while self.open_set and self.current_iteration < self.config['max_iterations']:
            self.current_iteration += 1
            
            # Get node with lowest cost
            current_cost, current_node = heapq.heappop(self.open_set)
            
            # Skip if already processed
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
                    algorithm_name="Dijkstra",
                    explored_nodes=self.explored_nodes.copy(),
                    final_cost=current_node.cost
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
                
                # Calculate tentative cost
                edge_cost = current_node.distance_to(neighbor)
                tentative_cost = current_node.cost + edge_cost
                
                # Check if this path to neighbor is better
                if (neighbor.id not in self.cost_so_far or 
                    tentative_cost < self.cost_so_far[neighbor.id]):
                    
                    # Update neighbor
                    neighbor.parent = current_node
                    neighbor.cost = tentative_cost
                    self.cost_so_far[neighbor.id] = tentative_cost
                    
                    # Add to open set
                    heapq.heappush(self.open_set, (tentative_cost, neighbor))
        
        # No path found
        return self.create_result(
            success=False,
            algorithm_name="Dijkstra",
            explored_nodes=self.explored_nodes.copy(),
            termination_reason="max_iterations" if self.current_iteration >= self.config['max_iterations'] else "no_path"
        )
    
    def _get_neighbors(self, node: Node) -> List[Node]:
        """
        Get neighboring nodes for expansion.
        Uses 8-connected grid (including diagonals).
        """
        neighbors = []
        step_size = self.config['step_size']
        
        # 8-connected neighbors (including diagonals)
        directions = [
            (step_size, 0),      # East
            (-step_size, 0),     # West  
            (0, step_size),      # North
            (0, -step_size),     # South
            (step_size, step_size),      # Northeast
            (-step_size, step_size),     # Northwest
            (step_size, -step_size),     # Southeast
            (-step_size, -step_size)     # Southwest
        ]
        
        for dx, dy in directions:
            new_position = node.position + np.array([dx, dy])
            
            # Check bounds
            if not self.environment.is_within_bounds(new_position):
                continue
            
            # Create neighbor node
            neighbor = Node(new_position, use_position_id=True)
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        return {
            'name': 'Dijkstra',
            'optimal': True,
            'complete': True,
            'space_complexity': 'O(V + E)',
            'time_complexity': 'O((V + E) log V)',
            'parameters': [
                'step_size', 'max_iterations', 'goal_tolerance', 'connection_radius'
            ],
            'description': 'Classic shortest path algorithm, guarantees optimal solution',
            'year': 1956,
            'category': 'Classical Graph-Based'
        }


# Example usage and test script
if __name__ == "__main__":
    # Test the complete system
    from environments.simple_2d import Simple2D
    from visualization.matplotlib_visualizer import MatplotlibVisualizer
    
    # Create environment
    env = Simple2D.empty_environment(size=20)
    start, goal = env.create_test_scenario("simple")
    
    # Create planner
    dijkstra = Dijkstra({
        'step_size': .5,
        'goal_tolerance': 1.0,
        'max_iterations': 5000
    })
    
    # Plan path
    print("Planning path with Dijkstra...")
    result = dijkstra.plan(start, goal, env)
    
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
                                  title="Dijkstra Algorithm Demo")
    
    # Show plot
    import matplotlib.pyplot as plt
    plt.show()
    
    # Save figure
    fig.savefig('dijkstra_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'dijkstra_demo.png'")