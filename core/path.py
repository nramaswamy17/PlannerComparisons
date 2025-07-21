# core/path.py
import numpy as np
from typing import List, Optional, Tuple
from .node import Node

class Path:
    """
    Represents a path through the planning space.
    Contains nodes, provides path analysis, and supports visualization.
    """
    def __init__(self, nodes: List[Node] = None):
        self.nodes = nodes or []
        self._length = None  # Cached length
        self._waypoints = None  # Cached waypoint positions
        
    @classmethod
    def from_node_chain(cls, end_node: Node) -> 'Path':
        """Create path by following parent pointers from end node"""
        node_chain = end_node.get_path_to_root()
        return cls(list(reversed(node_chain)))
    
    @classmethod
    def from_positions(cls, positions: List[np.ndarray]) -> 'Path':
        """Create path from list of positions"""
        nodes = []
        for i, pos in enumerate(positions):
            parent = nodes[-1] if nodes else None
            cost = 0.0 if not parent else parent.cost + np.linalg.norm(pos - parent.position)
            nodes.append(Node(pos, parent, cost))
        return cls(nodes)
    
    def add_node(self, node: Node):
        """Add a node to the end of the path"""
        if self.nodes:
            node.parent = self.nodes[-1]
            node.cost = self.nodes[-1].cost + self.nodes[-1].distance_to(node)
        self.nodes.append(node)
        self._invalidate_cache()
    
    def prepend_node(self, node: Node):
        """Add a node to the beginning of the path"""
        if self.nodes:
            self.nodes[0].parent = node
        self.nodes.insert(0, node)
        self._recompute_costs()
        self._invalidate_cache()
    
    def extend(self, other_path: 'Path'):
        """Extend this path with another path"""
        for node in other_path.nodes:
            self.add_node(Node(node.position.copy()))
    
    @property
    def start(self) -> Optional[Node]:
        """Start node of the path"""
        return self.nodes[0] if self.nodes else None
    
    @property
    def goal(self) -> Optional[Node]:
        """Goal node of the path"""
        return self.nodes[-1] if self.nodes else None
    
    @property
    def length(self) -> float:
        """Total path length"""
        if self._length is None:
            if len(self.nodes) < 2:
                self._length = 0.0
            else:
                self._length = sum(
                    self.nodes[i].distance_to(self.nodes[i+1])
                    for i in range(len(self.nodes) - 1)
                )
        return self._length
    
    @property
    def waypoints(self) -> np.ndarray:
        """Array of path waypoint positions"""
        if self._waypoints is None:
            if self.nodes:
                self._waypoints = np.array([node.position for node in self.nodes])
            else:
                self._waypoints = np.empty((0, 2))
        return self._waypoints
    
    @property
    def is_empty(self) -> bool:
        """Check if path is empty"""
        return len(self.nodes) == 0
    
    def get_segment(self, start_idx: int, end_idx: int) -> 'Path':
        """Get a segment of the path"""
        return Path(self.nodes[start_idx:end_idx])
    
    def interpolate(self, step_size: float) -> 'Path':
        """Create a new path with points interpolated at regular intervals"""
        if len(self.nodes) < 2:
            return Path(self.nodes.copy())
        
        interpolated_nodes = [Node(self.nodes[0].position.copy())]
        
        for i in range(len(self.nodes) - 1):
            start_pos = self.nodes[i].position
            end_pos = self.nodes[i + 1].position
            segment_vec = end_pos - start_pos
            segment_length = np.linalg.norm(segment_vec)
            
            if segment_length > step_size:
                num_steps = int(segment_length / step_size)
                for j in range(1, num_steps + 1):
                    alpha = j / num_steps
                    interp_pos = start_pos + alpha * segment_vec
                    interpolated_nodes.append(Node(interp_pos))
            
            # Always add the end node
            interpolated_nodes.append(Node(end_pos.copy()))
        
        return Path(interpolated_nodes)
    
    def smooth(self, window_size: int = 3) -> 'Path':
        """Apply simple moving average smoothing"""
        if len(self.nodes) <= window_size:
            return Path([Node(node.position.copy()) for node in self.nodes])
        
        smoothed_nodes = []
        half_window = window_size // 2
        
        # Keep start node unchanged
        smoothed_nodes.append(Node(self.nodes[0].position.copy()))
        
        # Smooth middle nodes
        for i in range(1, len(self.nodes) - 1):
            start_idx = max(0, i - half_window)
            end_idx = min(len(self.nodes), i + half_window + 1)
            
            positions = [self.nodes[j].position for j in range(start_idx, end_idx)]
            avg_position = np.mean(positions, axis=0)
            smoothed_nodes.append(Node(avg_position))
        
        # Keep goal node unchanged
        if len(self.nodes) > 1:
            smoothed_nodes.append(Node(self.nodes[-1].position.copy()))
        
        return Path(smoothed_nodes)
    
    def get_curvature_at(self, index: int) -> float:
        """Calculate curvature at a specific node index"""
        if index <= 0 or index >= len(self.nodes) - 1:
            return 0.0
        
        p1 = self.nodes[index - 1].position
        p2 = self.nodes[index].position
        p3 = self.nodes[index + 1].position
        
        # Use the formula for curvature of a discrete path
        v1 = p2 - p1
        v2 = p3 - p2
        
        cross_prod = np.cross(v1, v2)
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        
        if v1_mag == 0 or v2_mag == 0:
            return 0.0
        
        return abs(cross_prod) / (v1_mag * v2_mag)
    
    def get_max_curvature(self) -> float:
        """Get maximum curvature along the path"""
        if len(self.nodes) < 3:
            return 0.0
        
        return max(self.get_curvature_at(i) for i in range(1, len(self.nodes) - 1))
    
    def _recompute_costs(self):
        """Recompute costs for all nodes"""
        for i, node in enumerate(self.nodes):
            if i == 0:
                node.cost = 0.0
            else:
                node.cost = self.nodes[i-1].cost + self.nodes[i-1].distance_to(node)
    
    def _invalidate_cache(self):
        """Invalidate cached computations"""
        self._length = None
        self._waypoints = None
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __bool__(self) -> bool:
        return not self.is_empty
    
    def __repr__(self) -> str:
        return f"Path(nodes={len(self.nodes)}, length={self.length:.2f})"
