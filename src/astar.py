import heapq
import math
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

Cell = Tuple[int, int]

# Pre-computed neighbour offsets and their step costs
_NEIGHBOURS: Tuple[Tuple[int, int, float], ...] = (
    (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0,  1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1,  -1, math.sqrt(2)), (1,  1, math.sqrt(2)),
)

def is_within_bounds(cell: Cell, grid: np.ndarray) -> bool:
    r, c = cell
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def heuristic_formula(a: Cell, b: Cell) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

def neighbours(cell: Cell, grid: np.ndarray) -> Iterable[Tuple[Cell, float]]:
    r, c = cell
    for dr, dc, cost in _NEIGHBOURS:
        nb = (r + dr, c + dc)
        if not is_within_bounds(nb, grid):
            continue
        if grid[nb] == 1:
            continue
        # prevent going diagonally across two blocked cells
        if dr != 0 and dc != 0:
            if grid[r + dr, c] == 1 and grid[r, c + dc] == 1:
                continue
        yield nb, cost


def reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

###---------
##    Compute the shortest path of unblocked cells from `start` to `goal`.
##    Returns a list of cells [start, ..., goal] or None if no path exists.
###---------
def astar(grid: np.ndarray, start: Cell, goal: Cell) -> Optional[List[Cell]]:

    if not (is_within_bounds(start, grid) and is_within_bounds(goal, grid)):
        return None
    if grid[start] == 1 or grid[goal] == 1:
        return None
    if start == goal:
        return [start]

    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic_formula(start, goal), counter, start))

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    closed: set = set()

    while open_heap:
        popped_item = heapq.heappop(open_heap)
        current = popped_item[2]
        if current in closed:
            continue
        if current == goal:
            return reconstruct(came_from, current)
        closed.add(current)

        for nb, step in neighbours(current, grid):
            tentative = g_score[current] + step
            if tentative < g_score.get(nb, math.inf):
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + heuristic_formula(nb, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, nb))

    return None
