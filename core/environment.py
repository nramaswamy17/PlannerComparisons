# core/environment.py
import numpy as np
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from .node import Node
from .path import Path

class Obstacle:
    """Represents an obstacle in the environment"""
    def __init__(self, shape: str, params: dict):
        self.shape = shape  # 'circle', 'rectangle', 'polygon'
        self.params = params
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside this obstacle"""
        if self.shape == 'circle':
            center = np.array(self.params['center'])
            radius = self.params['radius']
            return np.linalg.norm(point - center) <= radius
        
        elif self.shape == 'rectangle':
            x, y = point
            x_min, y_min = self.params['bottom_left']
            x_max, y_max = self.params['top_right']
            return x_min <= x <= x_max and y_min <= y <= y_max
        
        elif self.shape == 'polygon':
            # Using ray casting algorithm for point-in-polygon
            vertices = np.array(self.params['vertices'])
            return self._point_in_polygon(point, vertices)
        
        return False
    
    def _point_in_polygon(self, point: np.ndarray, vertices: np.ndarray) -> bool:
        """Ray casting algorithm for point-in-polygon test"""
        x, y = point
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class BaseEnvironment(ABC):
    """Abstract base class for planning environments"""
    
    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        """
        Initialize environment with bounds.
        bounds: ((x_min, x_max), (y_min, y_max))
        """
        self.bounds = bounds
        self.x_bounds = bounds[0]
        self.y_bounds = bounds[1]
        self.obstacles: List[Obstacle] = []
        
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the environment"""
        self.obstacles.append(obstacle)
    
    def add_circle_obstacle(self, center: Tuple[float, float], radius: float):
        """Convenience method to add circular obstacle"""
        obstacle = Obstacle('circle', {'center': center, 'radius': radius})
        self.add_obstacle(obstacle)
    
    def add_rectangle_obstacle(self, bottom_left: Tuple[float, float], 
                             top_right: Tuple[float, float]):
        """Convenience method to add rectangular obstacle"""
        obstacle = Obstacle('rectangle', {
            'bottom_left': bottom_left,
            'top_right': top_right
        })
        self.add_obstacle(obstacle)
    
    def add_polygon_obstacle(self, vertices: List[Tuple[float, float]]):
        """Convenience method to add polygonal obstacle"""
        obstacle = Obstacle('polygon', {'vertices': vertices})
        self.add_obstacle(obstacle)
    
    @abstractmethod
    def is_collision_free(self, point: Union[Node, np.ndarray]) -> bool:
        """Check if a point is collision-free"""
        pass
    
    @abstractmethod
    def is_path_collision_free(self, path: Union[Path, List[Node], np.ndarray]) -> bool:
        """Check if a path is collision-free"""
        pass
    
    def is_within_bounds(self, point: Union[Node, np.ndarray]) -> bool:
        """Check if point is within environment bounds"""
        if isinstance(point, Node):
            pos = point.position
        else:
            pos = np.array(point)
        
        x, y = pos[0], pos[1]
        return (self.x_bounds[0] <= x <= self.x_bounds[1] and 
                self.y_bounds[0] <= y <= self.y_bounds[1])
    
    def sample_random_free_point(self) -> np.ndarray:
        """Sample a random collision-free point in the environment"""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
            y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
            point = np.array([x, y])
            
            if self.is_collision_free(point):
                return point
        
        raise RuntimeError("Could not sample free point after maximum attempts")
    
    def get_nearest_obstacle_distance(self, point: Union[Node, np.ndarray]) -> float:
        """Get distance to nearest obstacle (useful for potential fields)"""
        if isinstance(point, Node):
            pos = point.position
        else:
            pos = np.array(point)
        
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            if obstacle.shape == 'circle':
                center = np.array(obstacle.params['center'])
                radius = obstacle.params['radius']
                distance = max(0, np.linalg.norm(pos - center) - radius)
            
            elif obstacle.shape == 'rectangle':
                # Distance to rectangle boundary
                x, y = pos
                x_min, y_min = obstacle.params['bottom_left']
                x_max, y_max = obstacle.params['top_right']
                
                dx = max(0, max(x_min - x, x - x_max))
                dy = max(0, max(y_min - y, y - y_max))
                distance = np.sqrt(dx**2 + dy**2)
            
            else:  # polygon - approximate with centroid distance
                vertices = np.array(obstacle.params['vertices'])
                centroid = np.mean(vertices, axis=0)
                distance = np.linalg.norm(pos - centroid)
            
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def get_environment_info(self) -> dict:
        """Get information about the environment"""
        return {
            'bounds': self.bounds,
            'num_obstacles': len(self.obstacles),
            'obstacle_types': [obs.shape for obs in self.obstacles],
            'area': (self.x_bounds[1] - self.x_bounds[0]) * (self.y_bounds[1] - self.y_bounds[0])
        }


class Environment2D(BaseEnvironment):
    """Concrete 2D environment implementation"""
    
    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 resolution: float = 0.1):
        super().__init__(bounds)
        self.resolution = resolution
    
    def is_collision_free(self, point: Union[Node, np.ndarray]) -> bool:
        """Check if a point is collision-free"""
        if isinstance(point, Node):
            pos = point.position
        else:
            pos = np.array(point)
        
        # Check bounds
        if not self.is_within_bounds(pos):
            return False
        
        # Check obstacles
        for obstacle in self.obstacles:
            if obstacle.contains_point(pos):
                return False
        
        return True
    
    def is_path_collision_free(self, path: Union[Path, List[Node], np.ndarray]) -> bool:
        """Check if a path is collision-free by discretizing and checking points"""
        if isinstance(path, Path):
            waypoints = path.waypoints
        elif isinstance(path, list) and isinstance(path[0], Node):
            waypoints = np.array([node.position for node in path])
        else:
            waypoints = np.array(path)
        
        if len(waypoints) < 2:
            return self.is_collision_free(waypoints[0] if len(waypoints) == 1 else np.array([0, 0]))
        
        # Check each segment
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Discretize segment
            distance = np.linalg.norm(end - start)
            num_checks = max(2, int(distance / self.resolution))
            
            for j in range(num_checks + 1):
                alpha = j / num_checks
                point = start + alpha * (end - start)
                
                if not self.is_collision_free(point):
                    return False
        
        return True
    
    def get_grid_representation(self, resolution: Optional[float] = None) -> np.ndarray:
        """Get a grid representation of the environment (useful for visualization)"""
        if resolution is None:
            resolution = self.resolution
        
        x_range = self.x_bounds[1] - self.x_bounds[0]
        y_range = self.y_bounds[1] - self.y_bounds[0]
        
        nx = int(x_range / resolution) + 1
        ny = int(y_range / resolution) + 1
        
        grid = np.zeros((ny, nx))
        
        for i in range(ny):
            for j in range(nx):
                x = self.x_bounds[0] + j * resolution
                y = self.y_bounds[0] + i * resolution
                point = np.array([x, y])
                
                if not self.is_collision_free(point):
                    grid[i, j] = 1  # Obstacle
        
        return grid