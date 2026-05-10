

from __future__ import annotations
import math
from typing import Tuple
from osm_to_grid import GridSpec

EARTH_RADIUS_M = 6_371_000.0


# ------------
# Coordinate conversions
# ------------

# Map a GPS fix to a (row, col) cell, within grid bounds.
def gps_to_cell(lat: float, lon: float, spec: GridSpec) -> Tuple[int, int]:

    row = int((lat - spec.min_lat) / spec.lat_step)
    col = int((lon - spec.min_lon) / spec.lon_step)
    row = max(0, min(spec.rows - 1, row))
    col = max(0, min(spec.cols - 1, col))
    return row, col

# Return the GPS coordinates of the centre of the given cell
def cell_to_gps(row: int, col: int, spec: GridSpec) -> Tuple[float, float]:

    lat = spec.min_lat + (row + 0.5) * spec.lat_step
    lon = spec.min_lon + (col + 0.5) * spec.lon_step
    return lat, lon


# ------------
# Distance helpers (used by Kalman filter & sensor model)
# ------------
# Great-circle distance between two points in metres.
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


# Approximate ground size of one cell, returned as (height_m, width_m)
def cell_size_m(spec: GridSpec) -> Tuple[float, float]:

    mid_lat = 0.5 * (spec.min_lat + spec.max_lat)
    h = haversine_m(spec.min_lat, mid_lat, spec.min_lat + spec.lat_step, mid_lat)
    w = haversine_m(mid_lat, spec.min_lon, mid_lat, spec.min_lon + spec.lon_step)
    return h, w