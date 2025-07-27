# algorithms/classical/dstar.py
import heapq
import numpy as np
from typing import Dict, Any, List, Set, Optional
from enum import Enum
from core.base_planner import BasePlanner, PlanningResult
from core.environment import BaseEnvironment
from core.node import Node
from core.path import Path

class StateType(Enum):
    """State classifications for D* algorithm"""
    NEW = "NEW"      # Never been in open list
    OPEN = "OPEN"    # Currently in open list
    CLOSED = "CLOSED" # Removed from open list

class DStarState:
    """
    State representation for D* algorithm.
    Each state maintains cost information and classification.
    """
    def __init__(self, position: np.ndarray):
        self.position = position
        self.id = f"{position[0]:.3f},{position[1]:.3f}"  # Position-based ID like Dijkstra
        self.h = float('inf')  # Cost-to-goal estimate
        self.k = float('inf')  # Key value for priority queue
        self.state_type = StateType.NEW
        self.parent: Optional['DStarState'] = None
        self.neighbors: List['DStarState'] = []
        
    def __lt__(self, other):
        """Comparison for priority queue (based on k value)"""
        return self.k < other.k
        
    def __eq__(self, other):
        """Equality based on position"""
        return np.allclose(self.position, other.position, atol=1e-3)
        
    def distance_to(self, other: 'DStarState') -> float:
        """Calculate Euclidean distance to another state"""
        return np.linalg.norm(self.position - other.position)

class DStar(BasePlanner):
    """
    D* algorithm implementation for motion planning.
    Searches backwards from goal to compute optimal cost field,
    then extracts optimal path from start to goal.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize D* planner.
        
        Config options:
            - step_size: Step size for expanding nodes (default: 1.0)
            - max_iterations: Maximum iterations before giving up (default: 10000)
            - goal_tolerance: Distance tolerance for reaching goal (default: 0.5)
            - allow_diagonal: Allow diagonal movement (default: True)
            - connection_radius: Maximum distance to connect states (default: 2.0)
        """
        default_config = {
            'step_size': 1.0,
            'max_iterations': 10000,
            'goal_tolerance': 0.5,
            'allow_diagonal': True,
            'connection_radius': 2.0
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        self.open_list: List[DStarState] = []
        self.states: Dict[str, DStarState] = {}  # All states by ID
        self.goal_state: Optional[DStarState] = None
        self.start_state: Optional[DStarState] = None
    
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             environment: BaseEnvironment, **kwargs) -> PlanningResult:
        """
        Plan path using D* algorithm.
        
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
        self.open_list = []
        self.states = {}
        self.explored_nodes = []
        self.current_iteration = 0
        
        # Create goal and start states
        self.goal_state = self._get_or_create_state(goal)
        self.start_state = self._get_or_create_state(start)
        
        # Initialize goal state
        self.goal_state.h = 0.0
        self.goal_state.k = 0.0
        self.goal_state.state_type = StateType.OPEN
        heapq.heappush(self.open_list, self.goal_state)
        
        # Phase 1: Compute optimal costs from goal backwards
        print("Phase 1: Computing optimal cost field...")
        phase1_success = self._compute_optimal_costs()
        
        if not phase1_success:
            return self.create_result(
                success=False,
                algorithm_name="D*",
                explored_nodes=self.explored_nodes.copy(),
                termination_reason="failed_to_compute_costs"
            )
        
        # Phase 2: Extract optimal path from start to goal
        print("Phase 2: Extracting optimal path...")
        path_result = self._extract_path()
        
        if path_result is None:
            return self.create_result(
                success=False,
                algorithm_name="D*",
                explored_nodes=self.explored_nodes.copy(),
                termination_reason="no_path_found"
            )
        
        # Convert DStarStates to Nodes for path creation
        path_nodes = []
        current_state = self.start_state
        
        while current_state is not None:
            node = Node(current_state.position)
            node.cost = current_state.h
            path_nodes.append(node)
            current_state = current_state.parent
        
        # Create path from nodes
        if len(path_nodes) > 1:
            # Link nodes for path creation
            for i in range(len(path_nodes) - 1):
                path_nodes[i + 1].parent = path_nodes[i]
            
            path = Path.from_node_chain(path_nodes[-1])  # Start from goal node
            
            return self.create_result(
                path=path,
                success=True,
                algorithm_name="D*",
                explored_nodes=self.explored_nodes.copy(),
                final_cost=self.start_state.h,
                nodes_in_open_list=len([s for s in self.states.values() if s.state_type == StateType.OPEN]),
                nodes_in_closed_list=len([s for s in self.states.values() if s.state_type == StateType.CLOSED])
            )
        
        return self.create_result(
            success=False,
            algorithm_name="D*",
            explored_nodes=self.explored_nodes.copy(),
            termination_reason="path_extraction_failed"
        )
    
    def _compute_optimal_costs(self) -> bool:
        """
        Phase 1: Compute optimal costs using D* backward search.
        Returns True if start state cost was computed, False otherwise.
        """
        while (self.open_list and 
               self.current_iteration < self.config['max_iterations']):
            
            self.current_iteration += 1
            
            # Process minimum k-value state
            current_state = heapq.heappop(self.open_list)
            
            # Skip if already closed (outdated entry)
            if current_state.state_type == StateType.CLOSED:
                continue
            
            # Mark as closed and add to explored
            current_state.state_type = StateType.CLOSED
            self.explored_nodes.append(Node(current_state.position))
            
            # Check if we've reached the start state
            if np.allclose(current_state.position, self.start_state.position, 
                          atol=self.config['goal_tolerance']):
                self.start_state = current_state  # Update reference
                return True
            
            # Process neighbors
            neighbors = self._get_neighbors(current_state)
            
            for neighbor in neighbors:
                # Skip if not collision free
                neighbor_node = Node(neighbor.position)
                if not self.environment.is_collision_free(neighbor_node):
                    continue
                
                # Calculate edge cost
                edge_cost = current_state.distance_to(neighbor)
                new_cost = current_state.h + edge_cost
                
                # Update neighbor if we found a better path
                if neighbor.state_type == StateType.NEW:
                    # First time seeing this state
                    neighbor.h = new_cost
                    neighbor.parent = current_state
                    neighbor.k = new_cost
                    neighbor.state_type = StateType.OPEN
                    heapq.heappush(self.open_list, neighbor)
                    
                elif new_cost < neighbor.h:
                    # Found better path to existing state
                    neighbor.h = new_cost
                    neighbor.parent = current_state
                    
                    if neighbor.state_type == StateType.CLOSED:
                        # Reopen closed state
                        neighbor.k = new_cost
                        neighbor.state_type = StateType.OPEN
                        heapq.heappush(self.open_list, neighbor)
                    else:
                        # Update open state (add new entry, old will be ignored)
                        neighbor.k = new_cost
                        heapq.heappush(self.open_list, neighbor)
        
        # Check if start state was reached
        return (self.start_state.state_type != StateType.NEW and 
                self.start_state.h != float('inf'))
    
    def _extract_path(self) -> Optional[List[DStarState]]:
        """
        Phase 2: Extract optimal path from start to goal by following parent pointers.
        Returns list of states from start to goal, or None if no path.
        """
        if self.start_state.h == float('inf'):
            return None
        
        path = []
        current = self.start_state
        
        # Follow parent pointers to goal
        while current is not None:
            path.append(current)
            if np.allclose(current.position, self.goal_state.position, 
                          atol=self.config['goal_tolerance']):
                break
            current = current.parent
        
        return path if len(path) > 1 else None
    
    def _get_or_create_state(self, position: np.ndarray) -> DStarState:
        """Get existing state or create new one at position"""
        state_id = f"{position[0]:.3f},{position[1]:.3f}"
        
        if state_id not in self.states:
            self.states[state_id] = DStarState(position)
        
        return self.states[state_id]
    
    def _get_neighbors(self, state: DStarState) -> List[DStarState]:
        """
        Get neighboring states for expansion.
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
            new_position = state.position + np.array([dx, dy])
            
            # Check bounds
            if not self.environment.is_within_bounds(new_position):
                continue
            
            # Get or create neighbor state
            neighbor = self._get_or_create_state(new_position)
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information"""
        return {
            'name': 'D*',
            'optimal': True,
            'complete': True,
            'space_complexity': 'O(V)',
            'time_complexity': 'O(V log V + E)',
            'parameters': [
                'step_size', 'max_iterations', 'goal_tolerance', 
                'allow_diagonal', 'connection_radius'
            ],
            'description': 'Dynamic programming approach that searches backwards from goal',
            'year': 1994,
            'category': 'Classical Graph-Based',
            'advantages': ['Optimal', 'Complete', 'Efficient for replanning', 'Good for dynamic environments'],
            'disadvantages': ['More complex than A*', 'Higher memory usage', 'Overkill for static environments']
        }


# Example usage and test script for D*
if __name__ == "__main__":
    # Test the complete system for D*
    from environments.simple_2d import Simple2D
    from visualization.matplotlib_visualizer import MatplotlibVisualizer

    # Create environment
    env = Simple2D.with_simple_obstacles(size=20)
    start, goal = env.create_test_scenario("simple")

    # Create planner
    dstar = DStar({
        'step_size': 1.0,
        'goal_tolerance': 1.5,
        'max_iterations': 10000,
        'allow_diagonal': True
    })

    # Plan path
    print("Planning path with D*...")
    result = dstar.plan(start, goal, env)

    # Print results
    print(f"Planning result: {result}")
    if result.success:
        print(f"Path found! Length: {result.path.length:.2f}")
        print(f"Nodes explored: {result.nodes_explored}")
        print(f"Computation time: {result.computation_time:.3f} seconds")
        # D* specific info
        if hasattr(result, 'final_cost'):
            print(f"Final cost to start: {result.final_cost:.2f}")
        else:
            print(f"Final cost to start: {dstar.start_state.h:.2f}")
    else:
        print("No path found!")
        if hasattr(result, 'termination_reason'):
            print(f"Termination reason: {result.termination_reason}")

    # Visualize
    viz = MatplotlibVisualizer()
    fig = viz.plot_planning_result(result, env, start, goal,
                                   title="D* Algorithm Demo")

    # Show plot
    import matplotlib.pyplot as plt
    plt.show()

    # Save figure
    fig.savefig('dstar_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'dstar_demo.png'")