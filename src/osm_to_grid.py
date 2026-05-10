
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



    bbox = (spec.max_lat, spec.min_lat, spec.max_lon, spec.min_lon)
    tags = {"building": True, "natural": ["water", "wood"], "barrier": True}

    # geometries_from_bbox returns a GeoDataFrame of OSM features.
    gdf = ox.features_from_bbox(*bbox, tags=tags)

    grid = np.zeros((spec.rows, spec.cols), dtype=np.uint8)

    if gdf.empty:
        return grid

    # Rasterise: for every cell, test whether its centroid intersects
    # any OSM polygon. This is O(rows*cols*features) but with 6,400 cells
    # it remains comfortably fast for a single bbox.
    for r in range(spec.rows):
        cell_lat = spec.min_lat + (r + 0.5) * spec.lat_step
        for c in range(spec.cols):
            cell_lon = spec.min_lon + (c + 0.5) * spec.lon_step
            cell = box(
                cell_lon - spec.lon_step / 2,
                cell_lat - spec.lat_step / 2,
                cell_lon + spec.lon_step / 2,
                cell_lat + spec.lat_step / 2,
            )
            if gdf.intersects(cell).any():
                grid[r, c] = 1
    return grid


# ---------------
# Synthetic fallback (used by the demo when OSM is unreachable)
# --------------

def synthetic_grid(spec: GridSpec, seed: int = 7) -> np.ndarray:

    rng = np.random.default_rng(seed)
    grid = np.zeros((spec.rows, spec.cols), dtype=np.uint8)

    # one city block ~ 10 cells
    block = 10
    # 2-cell wide streets
    street = 2
    for r0 in range(0, spec.rows, block):
        for c0 in range(0, spec.cols, block):
            # leave the outer `street` rows/cols of each block empty
            r_start = r0 + street
            c_start = c0 + street
            r_end = min(r0 + block - street, spec.rows)
            c_end = min(c0 + block - street, spec.cols)
            if r_end <= r_start or c_end <= c_start:
                continue
            # 70% chance the block is built up, otherwise it's a park
            if rng.random() < 0.7:
                grid[r_start:r_end, c_start:c_end] = 1
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

    if use_osm:
        try:
            grid = osm_grid(spec)
            return grid, spec
        except Exception as exc:
            print(f"[osm_to_grid] OSM fetch failed ({exc!s}); using synthetic grid")

    grid = synthetic_grid(spec)
    return grid, spec