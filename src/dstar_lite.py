
from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

Cell = Tuple[int, int]
Key = Tuple[float, float]

INF = math.inf

_NEIGHBOURS: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
)


def _heuristic(a: Cell, b: Cell) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

# Incremental D* Lite planner over a binary occupancy grid
class DStarLite:

    MAX_EXPANSIONS = 200_000

    def __init__(self, grid: np.ndarray, start: Cell, goal: Cell) -> None:
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        self.start: Cell = start
        self.goal: Cell = goal
        self.km: float = 0.0

        self.g: Dict[Cell, float] = {}
        self.rhs: Dict[Cell, float] = {}
        self.U: List[Tuple[Key, int, Cell]] = []
        self.U_keys: Dict[Cell, Key] = {}
        self._counter = 0

        self.rhs[goal] = 0.0
        self._insert(goal, self._calc_key(goal))

    def _g(self, s: Cell) -> float:
        return self.g.get(s, INF)

    def _rhs(self, s: Cell) -> float:
        return self.rhs.get(s, INF)

    def _calc_key(self, s: Cell) -> Key:
        m = min(self._g(s), self._rhs(s))
        return (m + _heuristic(self.start, s) + self.km, m)

    def _insert(self, s: Cell, key: Key) -> None:
        self._counter += 1
        heapq.heappush(self.U, (key, self._counter, s))
        self.U_keys[s] = key

    def _remove(self, s: Cell) -> None:
        self.U_keys.pop(s, None)

    def _top(self) -> Tuple[Key, Cell]:
        while self.U:
            key, _, s = self.U[0]
            cur = self.U_keys.get(s)
            if cur is None or cur != key:
                heapq.heappop(self.U)
                continue
            return key, s
        return (INF, INF), (-1, -1)

    def _pop(self) -> Cell:
        while self.U:
            key, _, s = heapq.heappop(self.U)
            cur = self.U_keys.get(s)
            if cur is not None and cur == key:
                self.U_keys.pop(s, None)
                return s
        return (-1, -1)

    def _in_bounds(self, s: Cell) -> bool:
        return 0 <= s[0] < self.rows and 0 <= s[1] < self.cols

    def _succ(self, s: Cell) -> Iterable[Tuple[Cell, float]]:
        if self.grid[s] == 1:
            return
        r, c = s
        for dr, dc, cost in _NEIGHBOURS:
            nb = (r + dr, c + dc)
            if not self._in_bounds(nb):
                continue
            if self.grid[nb] == 1:
                continue
            if dr != 0 and dc != 0:
                if self.grid[r + dr, c] == 1 and self.grid[r, c + dc] == 1:
                    continue
            yield nb, cost

    _pred = _succ

    def _update_vertex(self, u: Cell) -> None:
        if u != self.goal:
            best = INF
            for nb, cost in self._succ(u):
                v = cost + self._g(nb)
                if v < best:
                    best = v
            self.rhs[u] = best
        if u in self.U_keys:
            self._remove(u)
        if self._g(u) != self._rhs(u):
            self._insert(u, self._calc_key(u))

    def _compute_shortest_path(self) -> None:
        expansions = 0
        while True:
            top_key, _ = self._top()
            start_key = self._calc_key(self.start)
            if top_key >= start_key and self._rhs(self.start) == self._g(self.start):
                return
            if expansions >= self.MAX_EXPANSIONS:
                return
            expansions += 1
            u = self._pop()
            if u == (-1, -1):
                return
            k_old = top_key
            k_new = self._calc_key(u)
            if k_old < k_new:
                self._insert(u, k_new)
            elif self._g(u) > self._rhs(u):
                self.g[u] = self._rhs(u)
                for nb, _ in self._pred(u):
                    self._update_vertex(nb)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for nb, _ in self._pred(u):
                    self._update_vertex(nb)

    def plan(self) -> Optional[List[Cell]]:
        self._compute_shortest_path()
        if self._g(self.start) == INF and self._rhs(self.start) == INF:
            return None
        path: List[Cell] = [self.start]
        current = self.start
        for _ in range(self.rows * self.cols + 1):
            if current == self.goal:
                return path
            best_cost = INF
            best_nb: Optional[Cell] = None
            for nb, cost in self._succ(current):
                total = cost + self._g(nb)
                if total < best_cost:
                    best_cost = total
                    best_nb = nb
            if best_nb is None or best_cost == INF:
                return None
            path.append(best_nb)
            current = best_nb
        return None

    def block_cell(self, cell: Cell) -> None:
        if not self._in_bounds(cell) or self.grid[cell] == 1:
            return
        self.grid[cell] = 1
        self.rhs[cell] = INF
        self.g[cell] = INF
        if cell in self.U_keys:
            self._remove(cell)
        for r in range(max(0, cell[0] - 1), min(self.rows, cell[0] + 2)):
            for c in range(max(0, cell[1] - 1), min(self.cols, cell[1] + 2)):
                self._update_vertex((r, c))

    def set_start(self, new_start: Cell) -> None:
        self.km += _heuristic(self.start, new_start)
        self.start = new_start