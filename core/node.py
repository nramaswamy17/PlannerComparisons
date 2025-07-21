# core/node.py
import numpy as np
from typing import Optional, List, Any
import uuid

class Node:
    """
    Represents a node/state in the planning space.
    Used by all motion planning algorithms for consistent state representation.
    """
    def __init__(self, position: np.ndarray, parent: Optional['Node'] = None, 
                 cost: float = 0.0, heuristic: float = 0.0, use_position_id: bool = False):
        
        if use_position_id:
            # Position-based ID for grid-based algorithms
            pos_rounded = np.round(position, 3)  # Round to avoid floating point issues
            self.id = f"{pos_rounded[0]:.3f},{pos_rounded[1]:.3f}"
        else:
            # UUID for sampling-based algorithms
            self.id = str(uuid.uuid4())[:8]  # Unique identifier
        
        self.position = np.array(position, dtype=float)
        self.parent = parent
        self.cost = cost  # g(n) - cost from start
        self.heuristic = heuristic  # h(n) - heuristic to goal
        self.children: List['Node'] = []
        
        # Algorithm-specific data can be stored here
        self.metadata = {}
        
    @property
    def f_score(self) -> float:
        """f(n) = g(n) + h(n) for A* and similar algorithms"""
        return self.cost + self.heuristic
    
    @property
    def x(self) -> float:
        """X coordinate"""
        return self.position[0]
    
    @property
    def y(self) -> float:
        """Y coordinate"""
        return self.position[1]
    
    @property
    def dimension(self) -> int:
        """Dimensionality of the node"""
        return len(self.position)
    
    def distance_to(self, other: 'Node') -> float:
        """Euclidean distance to another node"""
        return np.linalg.norm(self.position - other.position)
    
    def add_child(self, child: 'Node'):
        """Add a child node"""
        self.children.append(child)
        child.parent = self
    
    def get_path_to_root(self) -> List['Node']:
        """Get path from this node back to root (reversed)"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def __eq__(self, other: 'Node') -> bool:
        """Equality based on position (with small tolerance for floating point)"""
        if not isinstance(other, Node):
            return False
        return np.allclose(self.position, other.position, atol=1e-6)
    
    def __hash__(self) -> int:
        """Hash based on rounded position for use in sets/dicts"""
        return hash(tuple(np.round(self.position, 6)))
    
    def __lt__(self, other: 'Node') -> bool:
        """Less than comparison based on f_score (for priority queues)"""
        return self.f_score < other.f_score
    
    def __repr__(self) -> str:
        return f"Node(id={self.id}, pos={self.position}, cost={self.cost:.2f})"
