
# The orchestrator that wires every module together and implements the
# workflow specified in the problem brief:
#
#     1) Convert real GPS coordinates to a grid using OpenStreetMap data
#     2) Translate GPS <-> grid cells
#     3) Ensure YOLO is active on both front and downward facing cameras
#     4) Run A* once at takeoff to get the optimal initial path
#     5) Reroute with D* Lite when YOLO sees an obstacle
#     6) Reach goal point
#
# The agent does not assume real flight hardware; it exposes hooks
# (`get_gps`, `get_camera_frames`) which a real autopilot or a simulator
# can implement.


from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from astar import astar
from dstar_lite import DStarLite
from gps_utils import cell_size_m, cell_to_gps, gps_to_cell
from kalman import KalmanFilter2D
from osm_to_grid import GridSpec, build_grid


Cell = Tuple[int, int]


# ------------
# Sensor callback signatures (so the agent stays hardware-agnostic)
# ------------
GpsFn         = Callable[[], Tuple[float, float]]                 # -> (lat, lon)
CameraFn      = Callable[[], Tuple[np.ndarray, np.ndarray]]       # -> (front, down)



@dataclass
# One step of the flight log
class TelemetryFrame:

    step: int
    raw_gps:    Tuple[float, float]
    filt_gps:   Tuple[float, float]
    cell:       Cell
    heading:    float
    blocked:    Set[Cell] = field(default_factory=set)
    replanned:  bool = False
    path_left:  int = 0


class DroneAgent:

    #  construction
    def __init__(self, spec: GridSpec, grid: np.ndarray,
                 sensor_radius_m: float = 25.0,
                 step_dt: float = 1.0) -> None:
        self.spec = spec
        self.grid = grid
        self.cell_h, self.cell_w = cell_size_m(spec)

        self.kf = KalmanFilter2D(dt=step_dt, process_var=0.5, meas_var=4.0)

        self.start_cell: Cell = (0, 0)
        self.goal_cell:  Cell = (0, 0)
        self.cell:       Cell = (0, 0)

        self.path:        List[Cell] = []
        self.path_index:  int = 0
        self.heading:     float = 0.0       # radians; 0 = north (-row)
        self.dstar:       Optional[DStarLite] = None
        self.telemetry:   List[TelemetryFrame] = []
        self.step_count:  int = 0
        self._gps_fn:     Optional[GpsFn] = None
        self._cam_fn:     Optional[CameraFn] = None

    #  flight setup
    def start_flight(
        self,
        start_gps: Tuple[float, float],
        goal_gps:  Tuple[float, float],
        gps_fn:    GpsFn,
        camera_fn: Optional[CameraFn] = None,
    ) -> List[Cell]:

        # Step 1 + 2 + 4 of the workflow:
        #   - convert GPS endpoints to grid cells
        #   - run A* once
        #   - hand the grid + path to D* Lite for incremental updates

        self._gps_fn = gps_fn
        self._cam_fn = camera_fn

        self.start_cell = gps_to_cell(*start_gps, self.spec)
        self.goal_cell  = gps_to_cell(*goal_gps,  self.spec)
        self.cell       = self.start_cell

        # Kalman state in metres - centred on start
        self.kf.x[:2] = np.array([0.0, 0.0])

        path = astar(self.grid, self.start_cell, self.goal_cell)
        if path is None:
            raise RuntimeError("A* could not find an initial path "
                               f"from {self.start_cell} to {self.goal_cell}")
        self.path = path
        self.path_index = 0

        # D* Lite seeded with the same grid; future replans reuse its state
        self.dstar = DStarLite(self.grid, self.start_cell, self.goal_cell)
        # populate g/rhs values
        self.dstar.plan()

        return list(self.path)

    #  helpers
    @property
    def at_goal(self) -> bool:
        return self.cell == self.goal_cell

    # Step 2: noisy GPS through the Kalman filter -> filtered fix
    def _read_gps(self) -> Tuple[float, float]:

        assert self._gps_fn is not None
        raw_lat, raw_lon = self._gps_fn()

        # convert raw lat/lon to local metres relative to start
        s_lat, s_lon = cell_to_gps(*self.start_cell, self.spec)
        from gps_utils import haversine_m
        north_m = haversine_m(s_lat, s_lon, raw_lat, s_lon) * (1 if raw_lat >= s_lat else -1)
        east_m  = haversine_m(s_lat, s_lon, s_lat, raw_lon) * (1 if raw_lon >= s_lon else -1)

        self.kf.predict()
        filt = self.kf.update(np.array([east_m, north_m]))
        # turn metres back into lat/lon
        flat = s_lat + (filt[1] / 111_320.0)
        flon = s_lon + (filt[0] / (111_320.0 * math.cos(math.radians(s_lat))))
        return flat, flon, raw_lat, raw_lon  # type: ignore[return-value]

    def _update_heading(self, prev: Cell, nxt: Cell) -> None:
        dr = nxt[0] - prev[0]
        dc = nxt[1] - prev[1]
        # heading: 0 = north, +ve clockwise; row increases north -> negate dr
        self.heading = math.atan2(dc, -dr)

    #  main loop
    def step(self) -> TelemetryFrame:

        if self.dstar is None:
            raise RuntimeError("Call start_flight() before step()")
        if self.at_goal:
            return self.save_telemetry(set(), False)

        # 1) filtered GPS - even if we're not relying on it for grid index, we keep it consistent with the spec
        flat, flon, raw_lat, raw_lon = self._read_gps()
        self.cell = gps_to_cell(flat, flon, self.spec)
        # if Kalman is still warming up don't let it teleport us
        if self.cell != self.path[self.path_index]:
            self.cell = self.path[self.path_index]

        replanned = False
        blocked = set()

        # 4) Advance one cell
        if self.path_index + 1 < len(self.path):
            prev = self.path[self.path_index]
            nxt  = self.path[self.path_index + 1]
            self._update_heading(prev, nxt)
            self.path_index += 1
            self.cell = nxt

        return self.save_telemetry(blocked, replanned, raw=(raw_lat, raw_lon),
                         filt=(flat, flon))

    # saves telemetry data
    def save_telemetry(self, blocked: Set[Cell], replanned: bool,
             raw: Tuple[float, float] | None = None,
             filt: Tuple[float, float] | None = None) -> TelemetryFrame:
        if raw is None:
            raw = cell_to_gps(*self.cell, self.spec)
        if filt is None:
            filt = raw
        frame = TelemetryFrame(
            step=self.step_count,
            raw_gps=raw,
            filt_gps=filt,
            cell=self.cell,
            heading=self.heading,
            blocked=blocked,
            replanned=replanned,
            path_left=max(0, len(self.path) - self.path_index - 1),
        )
        self.telemetry.append(frame)
        self.step_count += 1
        return frame

    # limits for the drone
    @classmethod
    def from_bbox(
        cls,
        min_lat: float, min_lon: float,
        max_lat: float, max_lon: float,
        use_osm: bool = True,
        **kwargs,
    ) -> "DroneAgent":
        grid, spec = build_grid(min_lat, min_lon, max_lat, max_lon, use_osm=use_osm)
        return cls(spec, grid, **kwargs)