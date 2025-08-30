import numpy as np
from typing import List

class GridStrategy:
    """
    Calculates the desired grid of orders based on strategy parameters.
    """
    def __init__(self,
                 grid_range_low: float,
                 grid_range_high: float,
                 num_grids: int):
        """
        Initializes the grid strategy parameters.
        :param grid_range_low: The lower price boundary of the grid.
        :param grid_range_high: The upper price boundary of the grid.
        :param num_grids: The number of grid lines (levels).
        """
        if not grid_range_low < grid_range_high:
            raise ValueError("grid_range_low must be less than grid_range_high.")
        if num_grids < 2:
            raise ValueError("num_grids must be at least 2.")

        self.grid_range_low = grid_range_low
        self.grid_range_high = grid_range_high
        self.num_grids = num_grids
        print(f"GridStrategy initialized for range ${grid_range_low:,.2f} - ${grid_range_high:,.2f} with {num_grids} levels.")

    def generate_grid_levels(self) -> List[float]:
        """
        Generates the price levels for the grid.
        Uses a linear spacing between the low and high range.
        """
        return np.linspace(self.grid_range_low, self.grid_range_high, self.num_grids).tolist()

# Example Usage
if __name__ == '__main__':
    print("--- Grid Strategy Demonstration ---")
    try:
        # Example: A grid for an asset between $50,000 and $60,000 with 5 levels
        strategy = GridStrategy(grid_range_low=50000.0, grid_range_high=60000.0, num_grids=5)

        levels = strategy.generate_grid_levels()

        print(f"\nGenerated {len(levels)} grid levels:")
        for i, level in enumerate(levels):
            print(f"Level {i+1}: ${level:,.2f}")

    except ValueError as e:
        print(f"Error: {e}")
