# environments/simple_2d.py
import numpy as np
from typing import Tuple, List, Optional, Union
from core.environment import Environment2D
from core.node import Node
from core.path import Path

class Simple2D(Environment2D):
    """
    Simple 2D environment with optional basic obstacles.
    Perfect for testing algorithms and getting started.
    """
    
    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-10, 10), (-10, 10)),
                 resolution: float = 0.1):
        """
        Initialize simple 2D environment.
        
        Args:
            bounds: ((x_min, x_max), (y_min, y_max))
            resolution: Resolution for collision checking discretization
        """
        super().__init__(bounds, resolution)
        self.name = "Simple2D"
    
    @classmethod
    def empty_environment(cls, size: float = 20.0) -> 'Simple2D':
        """Create an empty square environment"""
        half_size = size / 2
        bounds = ((-half_size, half_size), (-half_size, half_size))
        return cls(bounds)
    
    @classmethod
    def with_simple_obstacles(cls, size: float = 20.0) -> 'Simple2D':
        """Create environment with a few simple obstacles for testing"""
        env = cls.empty_environment(size)
        
        # Add a few obstacles to make it interesting
        env.add_circle_obstacle(center=(2, 3), radius=1.5)
        env.add_rectangle_obstacle(bottom_left=(-4, -2), top_right=(-1, 1))
        env.add_circle_obstacle(center=(6, -4), radius=1.0)
        env.add_rectangle_obstacle(bottom_left=(3, 5), top_right=(7, 8))
        
        return env
    
    @classmethod
    def corridor_environment(cls) -> 'Simple2D':
        """Create a corridor-like environment"""
        env = cls(bounds=((-5, 15), (-3, 3)))
        
        # Create corridor walls
        env.add_rectangle_obstacle(bottom_left=(-5, 2), top_right=(15, 3))
        env.add_rectangle_obstacle(bottom_left=(-5, -3), top_right=(15, -2))
        
        # Add some obstacles in the corridor
        env.add_circle_obstacle(center=(3, 0), radius=0.8)
        env.add_circle_obstacle(center=(8, 0.5), radius=0.6)
        env.add_rectangle_obstacle(bottom_left=(12, -1), top_right=(13, 1))
        
        return env
    
    def create_test_scenario(self, scenario: str = "empty") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create predefined start/goal scenarios for testing.
        
        Args:
            scenario: "empty", "simple", "corridor", or "challenging"
            
        Returns:
            (start_position, goal_position)
        """
        if scenario == "empty":
            start = np.array([-8, -8])
            goal = np.array([8, 8])
        
        elif scenario == "simple":
            start = np.array([-8, -5])
            goal = np.array([8, 5])
        
        elif scenario == "corridor":
            start = np.array([-4, 0])
            goal = np.array([14, 0])
        
        elif scenario == "challenging":
            start = np.array([-8, -8])
            goal = np.array([8, 8])
            # This will be used with the simple obstacles environment
        
        else:
            # Random start and goal
            start = self.sample_random_free_point()
            goal = self.sample_random_free_point()
        
        # Ensure start and goal are valid
        if not self.is_collision_free(start):
            start = self.sample_random_free_point()
        if not self.is_collision_free(goal):
            goal = self.sample_random_free_point()
        
        return start, goal
