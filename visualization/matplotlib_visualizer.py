
# visualization/matplotlib_visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from core.environment import BaseEnvironment, Obstacle
from core.node import Node
from core.path import Path
from core.base_planner import PlanningResult

class MatplotlibVisualizer:
    """
    Matplotlib-based visualizer for motion planning results.
    Creates publication-quality plots for your portfolio.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')  # Fallback if style not available
        
        # Color scheme - professional looking
        self.colors = {
            'obstacle': '#2C3E50',
            'start': '#E74C3C',
            'goal': '#27AE60',
            'path': '#3498DB',
            'explored': '#95A5A6',
            'background': '#FFFFFF',
            'grid': '#ECF0F1'
        }
    
    def plot_environment(self, environment: BaseEnvironment, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot the environment with obstacles"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set bounds
        x_bounds, y_bounds = environment.bounds
        ax.set_xlim(x_bounds[0] - 1, x_bounds[1] + 1)
        ax.set_ylim(y_bounds[0] - 1, y_bounds[1] + 1)
        
        # Plot obstacles
        for obstacle in environment.obstacles:
            self._plot_obstacle(obstacle, ax)
        
        # Styling
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_facecolor(self.colors['background'])
        
        return ax
    
    def plot_planning_result(self, result: PlanningResult, environment: BaseEnvironment,
                           start: np.ndarray, goal: np.ndarray,
                           show_explored: bool = True, show_path: bool = True,
                           title: Optional[str] = None) -> plt.Figure:
        """
        Plot complete planning result.
        
        Args:
            result: Planning result from algorithm
            environment: Environment used for planning
            start: Start position
            goal: Goal position
            show_explored: Whether to show explored nodes
            show_path: Whether to show final path
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot environment
        self.plot_environment(environment, ax)
        
        # Plot explored nodes
        if show_explored and hasattr(result, 'metadata') and 'explored_nodes' in result.metadata:
            explored_nodes = result.metadata['explored_nodes']
            if explored_nodes:
                explored_positions = np.array([node.position for node in explored_nodes])
                ax.scatter(explored_positions[:, 0], explored_positions[:, 1],
                          c=self.colors['explored'], s=10, alpha=0.6, 
                          label=f'Explored ({len(explored_nodes)} nodes)', zorder=2)
        
        # Plot path
        if show_path and result.path and result.success:
            self.plot_path(result.path, ax, label=f'Path (length: {result.path.length:.2f})')
        
        # Plot start and goal
        ax.scatter(*start, c=self.colors['start'], s=200, marker='o', 
                  edgecolors='white', linewidth=2, label='Start', zorder=5)
        ax.scatter(*goal, c=self.colors['goal'], s=600, marker='*', 
                  edgecolors='white', linewidth=2, label='Goal', zorder=5)
        
        # Title and labels
        if title is None:
            if hasattr(result, 'metadata') and 'algorithm_name' in result.metadata:
                alg_name = result.metadata['algorithm_name']
            else:
                alg_name = "Motion Planning"
            
            status = "SUCCESS" if result.success else "FAILED"
            title = f"{alg_name} - {status} (Time: {result.computation_time:.3f}s)"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_path(self, path: Path, ax: Optional[plt.Axes] = None, 
                  color: Optional[str] = None, linewidth: float = 3,
                  label: Optional[str] = None, show_waypoints: bool = False) -> plt.Axes:
        """Plot a path"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if not path or len(path.nodes) == 0:
            return ax
        
        color = color or self.colors['path']
        waypoints = path.waypoints
        
        # Plot path line
        ax.plot(waypoints[:, 0], waypoints[:, 1], 
               color=color, linewidth=linewidth, label=label, zorder=3)
        
        # Plot waypoints if requested
        if show_waypoints:
            ax.scatter(waypoints[:, 0], waypoints[:, 1], 
                      c=color, s=30, alpha=0.8, zorder=4)
        
        return ax
    
    def plot_multiple_paths(self, paths: List[Tuple[Path, str]], environment: BaseEnvironment,
                           start: np.ndarray, goal: np.ndarray,
                           title: str = "Path Comparison") -> plt.Figure:
        """Plot multiple paths for comparison"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot environment
        self.plot_environment(environment, ax)
        
        # Plot paths with different colors
        colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C']
        
        for i, (path, label) in enumerate(paths):
            if path and len(path.nodes) > 0:
                color = colors[i % len(colors)]
                self.plot_path(path, ax, color=color, 
                             label=f"{label} (len: {path.length:.2f})")
        
        # Plot start and goal
        ax.scatter(*start, c=self.colors['start'], s=200, marker='o',
                  edgecolors='white', linewidth=2, label='Start', zorder=5)
        ax.scatter(*goal, c=self.colors['goal'], s=200, marker='*',
                  edgecolors='white', linewidth=2, label='Goal', zorder=5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_algorithm_comparison(self, results: List[Tuple[PlanningResult, str]],
                                environment: BaseEnvironment, start: np.ndarray, goal: np.ndarray) -> plt.Figure:
        """Create a comprehensive comparison plot"""
        fig = plt.figure(figsize=(16, 10))
        
        # Main plot with all paths
        ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        
        # Plot environment
        self.plot_environment(environment, ax_main)
        
        # Plot all successful paths
        colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C']
        path_data = []
        
        for i, (result, name) in enumerate(results):
            if result.success and result.path:
                color = colors[i % len(colors)]
                self.plot_path(result.path, ax_main, color=color, 
                             label=f"{name} ({result.path.length:.2f})")
                path_data.append((name, result))
        
        # Start and goal
        ax_main.scatter(*start, c=self.colors['start'], s=200, marker='o',
                       edgecolors='white', linewidth=2, label='Start', zorder=5)
        ax_main.scatter(*goal, c=self.colors['goal'], s=200, marker='*',
                       edgecolors='white', linewidth=2, label='Goal', zorder=5)
        
        ax_main.set_title('Algorithm Comparison', fontsize=14, fontweight='bold')
        ax_main.legend()
        
        # Performance comparison charts
        if path_data:
            # Computation time comparison
            ax_time = plt.subplot2grid((2, 3), (0, 2))
            names = [name for name, _ in path_data]
            times = [result.computation_time for _, result in path_data]
            
            bars = ax_time.bar(names, times, color=[colors[i % len(colors)] for i in range(len(names))])
            ax_time.set_title('Computation Time')
            ax_time.set_ylabel('Time (seconds)')
            plt.setp(ax_time.get_xticklabels(), rotation=45)
            
            # Path length comparison
            ax_length = plt.subplot2grid((2, 3), (1, 2))
            lengths = [result.path.length for _, result in path_data]
            
            bars = ax_length.bar(names, lengths, color=[colors[i % len(colors)] for i in range(len(names))])
            ax_length.set_title('Path Length')
            ax_length.set_ylabel('Length')
            plt.setp(ax_length.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _plot_obstacle(self, obstacle: Obstacle, ax: plt.Axes):
        """Plot a single obstacle"""
        if obstacle.shape == 'circle':
            center = obstacle.params['center']
            radius = obstacle.params['radius']
            circle = patches.Circle(center, radius, 
                                   facecolor=self.colors['obstacle'], 
                                   edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(circle)
        
        elif obstacle.shape == 'rectangle':
            bottom_left = obstacle.params['bottom_left']
            top_right = obstacle.params['top_right']
            width = top_right[0] - bottom_left[0]
            height = top_right[1] - bottom_left[1]
            
            rect = patches.Rectangle(bottom_left, width, height,
                                   facecolor=self.colors['obstacle'],
                                   edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(rect)
        
        elif obstacle.shape == 'polygon':
            vertices = obstacle.params['vertices']
            polygon = patches.Polygon(vertices,
                                    facecolor=self.colors['obstacle'],
                                    edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(polygon)
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """Save figure with high quality"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
