import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── Grid Configuration ────────────────────────────────────────────────────────
GRID_SIZE = 50          # 50x50 cells
CELL_SIZE = 5           # each cell = 5 meters in real world
BUILDING_COUNT = 15     # number of buildings to place
FREE = 0                # passable cell
BLOCKED = 1             # obstacle cell


# ── Grid Generator ────────────────────────────────────────────────────────────
def generate_grid(size=GRID_SIZE, building_count=BUILDING_COUNT, seed=42):
    """
    Generates a city-like grid with rectangular buildings.

    Args:
        size:           Grid dimensions (size x size)
        building_count: Number of buildings to place
        seed:           Random seed for reproducibility

    Returns:
        2D numpy array — 0 = free, 1 = blocked
    """
    random.seed(seed)
    grid = np.zeros((size, size), dtype=int)

    for _ in range(building_count):
        # Random building size between 3x3 and 10x10 cells
        w = random.randint(3, 10)
        h = random.randint(3, 10)

        # Random position — keep away from edges
        x = random.randint(2, size - w - 2)
        y = random.randint(2, size - h - 2)

        grid[y:y+h, x:x+w] = BLOCKED

    return grid


def get_free_cell(grid, exclude=None):
    """Returns a random free cell, optionally excluding a specific cell."""
    size = grid.shape[0]
    while True:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if grid[y][x] == FREE:
            if exclude is None or (x, y) != exclude:
                return (x, y)


# ── A* Pathfinding ────────────────────────────────────────────────────────────
def heuristic(a, b):
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    """
    A* pathfinding algorithm.

    Args:
        grid:   2D numpy array (0=free, 1=blocked)
        start:  (x, y) tuple
        goal:   (x, y) tuple

    Returns:
        List of (x, y) tuples from start to goal, or None if no path exists.
    """
    size = grid.shape[0]
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        x, y = current
        # Check all 4 neighbors (up, down, left, right)
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] == FREE:
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


# ── D* Lite Replanning ────────────────────────────────────────────────────────
def replan(grid, current_pos, goal, blocked_cell):
    """
    Replans route when an obstacle is detected.
    Marks the blocked cell and reruns A* from current position.

    Args:
        grid:         Current grid
        current_pos:  Drone's current (x, y)
        goal:         Destination (x, y)
        blocked_cell: The cell that was found to be blocked

    Returns:
        New path from current_pos to goal, or None if no path exists.
    """
    bx, by = blocked_cell
    grid[by][bx] = BLOCKED  # mark the newly discovered obstacle
    return astar(grid, current_pos, goal)


# ── Visualization ─────────────────────────────────────────────────────────────
def visualize(grid, path, start, goal, title="Drone Path Simulation"):
    """Renders the grid with buildings, path, start and goal."""
    fig, ax = plt.subplots(figsize=(10, 10))
    size = grid.shape[0]

    # Draw grid cells
    for y in range(size):
        for x in range(size):
            if grid[y][x] == BLOCKED:
                color = '#2c3e50'  # dark = building
            else:
                color = '#ecf0f1'  # light = free space
            rect = patches.Rectangle((x, y), 1, 1,
                                      linewidth=0.2,
                                      edgecolor='#bdc3c7',
                                      facecolor=color)
            ax.add_patch(rect)

    # Draw path
    if path:
        px = [p[0] + 0.5 for p in path]
        py = [p[1] + 0.5 for p in path]
        ax.plot(px, py, color='#e74c3c', linewidth=2, zorder=3, label='Path')

    # Draw start and goal
    ax.plot(start[0] + 0.5, start[1] + 0.5, 'go', markersize=12,
            zorder=4, label='Start (A)')
    ax.plot(goal[0] + 0.5, goal[1] + 0.5, 'r*', markersize=14,
            zorder=4, label='Goal (B)')

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f'X (1 cell = {CELL_SIZE}m)')
    ax.set_ylabel(f'Y (1 cell = {CELL_SIZE}m)')
    plt.tight_layout()
    plt.savefig('grid_simulation.png', dpi=150)
    plt.show()
    print("Grid saved to grid_simulation.png")


# ── Main Simulation ───────────────────────────────────────────────────────────
def simulate(start=None, goal=None):
    """
    Runs the full drone path simulation.

    Args:
        start: (x, y) or None for random
        goal:  (x, y) or None for random
    """
    # Generate city grid
    grid = generate_grid()

    # Place start and goal
    if start is None:
        start = get_free_cell(grid)
    if goal is None:
        goal = get_free_cell(grid, exclude=start)

    print(f"Grid size:  {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE * CELL_SIZE}m x {GRID_SIZE * CELL_SIZE}m)")
    print(f"Start (A):  {start}")
    print(f"Goal  (B):  {goal}")

    # Plan initial route with A*
    path = astar(grid, start, goal)

    if path is None:
        print("No path found — try a different grid seed or start/goal positions.")
        return

    print(f"A* path found: {len(path)} steps "
          f"({len(path) * CELL_SIZE}m total distance)")

    # Visualize
    visualize(grid, path, start, goal, title="Drone Simulation — A* Path")

    return grid, path, start, goal


if __name__ == "__main__":
    # Fixed start and goal for consistent testing
    # Swap for simulate() to randomize
    simulate(start=(2, 2), goal=(47, 47))