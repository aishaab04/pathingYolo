
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import osmnx as ox
from shapely.geometry import box, Polygon, MultiPolygon
import numpy as np

# This program converts a real-world bounding box of OpenStreetMap data into an 80 x 80
# occupancy grid where every cell is either 0  -> free / flyable airspace or  1  -> blocked  (building footprint, walls, large structure, etc.)
# The grid is the planning surface for A* and D* Lite.

GRID_SIZE = 80  # 80 x 80 grid


@dataclass
class GridSpec:

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    rows: int = GRID_SIZE
    cols: int = GRID_SIZE

    @property
    def lat_step(self) -> float:
        return (self.max_lat - self.min_lat) / self.rows

    @property
    def lon_step(self) -> float:
        return (self.max_lon - self.min_lon) / self.cols


# -----------
# Real Open Street Map ingestion
# -----------

def osm_grid(spec: GridSpec) -> np.ndarray:
    tags = {"building": True}

    # use features_from_polygon to avoid bbox NaN bug
    from shapely.geometry import box as shapely_box
    bbox_polygon = shapely_box(spec.min_lon, spec.min_lat, spec.max_lon, spec.max_lat)
    gdf = ox.features_from_polygon(bbox_polygon, tags=tags)

    grid = np.zeros((spec.rows, spec.cols), dtype=np.uint8)

    if gdf.empty:
        return grid

    for r in range(spec.rows):
        cell_lat = spec.min_lat + (r + 0.5) * spec.lat_step
        for c in range(spec.cols):
            cell_lon = spec.min_lon + (c + 0.5) * spec.lon_step
            
            # use center point only instead of full cell box
            from shapely.geometry import Point
            center = Point(cell_lon, cell_lat)
            if gdf.contains(center).any():
                grid[r, c] = 1
    return grid



# --------------
# Public entry point
# -------------

def build_grid(
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        use_osm: bool = True,
) -> Tuple[np.ndarray, GridSpec]:

    spec = GridSpec(min_lat, min_lon, max_lat, max_lon)
    grid = osm_grid(spec)
    return grid, spec
    