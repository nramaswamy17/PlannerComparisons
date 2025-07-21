
# core/base_planner.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
from .environment import BaseEnvironment
from .node import Node
from .path import Path
import numpy as np

class PlanningResult:
    """Container for planning results and metadata"""
    def __init__(self, path: Optional[Path] = None, success: bool = False,
                 computation_time: float = 0.0, nodes_explored: int = 0,
                 path_cost: float = float('inf'), iterations: int = 0,
                 metadata: Dict[str, Any] = None):
        self.path = path
        self.success = success
        self.computation_time = computation_time
        self.nodes_explored = nodes_explored
        self.path_cost = path_cost
        self.iterations = iterations
        self.metadata = metadata or {}
        
        # Derived metrics
        self.path_length = path.length if path else float('inf')
        self.planning_rate = nodes_explored / max(computation_time, 1e-6)  # nodes/second
    
    def __repr__(self) -> str:
        return (f"PlanningResult(success={self.success}, "
                f"time={self.computation_time:.3f}s, "
                f"nodes={self.nodes_explored}, "
                f"length={self.path_length:.2f})")


class BasePlanner(ABC):
    """Abstract base class for all motion planning algorithms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        
        # Planning state
        self.start_node: Optional[Node] = None
        self.goal_node: Optional[Node] = None
        self.environment: Optional[BaseEnvironment] = None
        
        # Results tracking
        self.explored_nodes: List[Node] = []
        self.current_iteration = 0
        self.start_time = 0.0
        
    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             environment: BaseEnvironment, **kwargs) -> PlanningResult:
        """
        Plan a path from start to goal in the given environment.
        
        Args:
            start: Start position as numpy array
            goal: Goal position as numpy array  
            environment: Environment to plan in
            **kwargs: Algorithm-specific parameters
            
        Returns:
            PlanningResult containing path and metadata
        """
        pass
    
    @abstractmethod
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about this algorithm.
        
        Returns:
            Dictionary with algorithm metadata including:
            - name: Algorithm name
            - optimal: Whether algorithm guarantees optimal paths
            - complete: Whether algorithm is probabilistically/resolution complete
            - space_complexity: Space complexity description
            - time_complexity: Time complexity description
            - parameters: List of important parameters
        """
        pass
    
    def setup_planning(self, start: np.ndarray, goal: np.ndarray, 
                      environment: BaseEnvironment):
        """Common setup for planning algorithms"""
        self.start_node = Node(start)
        self.goal_node = Node(goal)
        self.environment = environment
        self.explored_nodes = []
        self.current_iteration = 0
        self.start_time = time.time()
        
        # Validate start and goal
        if not environment.is_collision_free(self.start_node):
            raise ValueError("Start position is not collision-free")
        if not environment.is_collision_free(self.goal_node):
            raise ValueError("Goal position is not collision-free")
    
    def create_result(self, path: Optional[Path] = None, 
                     success: bool = False, **kwargs) -> PlanningResult:
        """Create a planning result with common metrics"""
        computation_time = time.time() - self.start_time
        
        return PlanningResult(
            path=path,
            success=success,
            computation_time=computation_time,
            nodes_explored=len(self.explored_nodes),
            path_cost=path.length if path else float('inf'),
            iterations=self.current_iteration,
            metadata=kwargs
        )
    
    def is_goal_reached(self, node: Node, tolerance: float = 0.1) -> bool:
        """Check if a node is close enough to the goal"""
        return node.distance_to(self.goal_node) <= tolerance
    
    def get_heuristic(self, node: Node, goal: Node, heuristic_type: str = 'euclidean') -> float:
        """Calculate heuristic distance between two nodes"""
        if heuristic_type == 'euclidean':
            return node.distance_to(goal)
        elif heuristic_type == 'manhattan':
            return np.sum(np.abs(node.position - goal.position))
        elif heuristic_type == 'octile':
            # Octile distance (allows diagonal movement)
            dx = abs(node.position[0] - goal.position[0])
            dy = abs(node.position[1] - goal.position[1])
            return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
        else:
            return node.distance_to(goal)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            'algorithm': self.name,
            'config': self.config.copy()
        }
    
    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"