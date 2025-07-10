import numpy as np
import matplotlib.pyplot as plt

def generate_dynamic_scenario(size=(50, 50), obstacle_density=0.1, start=(0,0), goal=(49,49), seed = 1):
    np.random.seed(seed)  # Set seed for reproducibility
    grid = np.zeros(size)
    obstacle_positions = np.random.rand(*size) < obstacle_density
    grid[obstacle_positions] = 1
    grid[start] = grid[goal] = 0  # Ensure start and goal are free
    return grid

if __name__ == "__main__":
    scenario = generate_dynamic_scenario()
    plt.imshow(scenario, cmap='Greys')
    plt.title("Sample Scenario")
    plt.show()
