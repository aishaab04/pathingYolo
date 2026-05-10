
# End-to-end demo of the delivery-drone navigation agent.
#
# * Builds an 80x80 grid for a small bounding box (San Francisco SoMa).
# * Picks a start and goal GPS pair near opposite corners.
# * Runs A* once.
# * Walks the drone forward, injecting two surprise obstacles partway
#   through the flight to trigger D* Lite re-routing.
# * Optionally writes a matplotlib PNG showing the original path,
#   obstacle locations, and the final actually-flown trajectory.

from obstacle import detect_obstacle
from matplotlib.animation import FuncAnimation

import os
import random
from typing import List, Tuple

import numpy as np
from animation import visualize

from drone_agent import DroneAgent
from gps_utils import cell_to_gps, gps_to_cell
import osmnx as ox

# convert addresses to GPS coordinates
START_ADDRESS = "175 Lindbergh Blvd, NJ, Teaneck"
GOAL_ADDRESS  = "320 Fabry Ter, NJ, Teaneck"

START_GPS = ox.geocode(START_ADDRESS)
GOAL_GPS  = ox.geocode(GOAL_ADDRESS)

# bounding box — automatically built around the two points
MIN_LAT = min(START_GPS[0], GOAL_GPS[0]) - 0.005
MAX_LAT = max(START_GPS[0], GOAL_GPS[0]) + 0.005
MIN_LON = min(START_GPS[1], GOAL_GPS[1]) - 0.005
MAX_LON = max(START_GPS[1], GOAL_GPS[1]) + 0.005

path_snapshots = []  # stores the path at each step
reroute_steps = []   # stores which steps had a reroute


# simulator Pretends to be a flight stack so the agent has things to read
class FakeWorld:

    def __init__(self, agent: DroneAgent, jitter_m: float = 3.0) -> None:
        self.agent = agent
        self.jitter_m = jitter_m
        self.rng = random.Random(42)
        self.step = 0               # ← this was missing
        self.camera_schedule = {}   # ← and this
    def gps(self) -> Tuple[float, float]:
        lat, lon = cell_to_gps(*self.agent.cell, self.agent.spec)
        # add Gaussian noise (~jitter_m metres) - the Kalman filter will smooth it
        dlat = self.rng.gauss(0, self.jitter_m / 111_320.0)
        dlon = self.rng.gauss(0, self.jitter_m / 95_000.0)
        return lat + dlat, lon + dlon

    def frames(self):
        front = None

        if self.step in self.camera_schedule:
            path = self.camera_schedule[self.step]
            print(f"\n[CAMERA] Step {self.step}: about to load → {path}")
            front = path  # pass path string directly to YOLO
            print(f"\n[CAMERA] Step {self.step}: captured → {os.path.basename(path)}")

        return front, None


from obstacle import detect_obstacle
from astar import astar

def run_demo(use_osm: bool = False, draw: bool = True) -> None:
    print("=== Delivery Drone AI Agent demo ===")
    print(f"Bounding box: ({MIN_LAT}, {MIN_LON}) -> ({MAX_LAT}, {MAX_LON})")

    agent = DroneAgent.from_bbox(
    MIN_LAT, MIN_LON, MAX_LAT, MAX_LON,
    sensor_radius_m=30.0,
)
    print(f"Grid built: {agent.grid.shape}  blocked cells = {int(agent.grid.sum())}")

    sim = FakeWorld(agent)

    start_cell = gps_to_cell(*START_GPS, agent.spec)
    goal_cell  = gps_to_cell(*GOAL_GPS,  agent.spec)
    start_cell = nearest_free(agent.grid, start_cell)
    goal_cell  = nearest_free(agent.grid, goal_cell)
    start_gps  = cell_to_gps(*start_cell, agent.spec)
    goal_gps   = cell_to_gps(*goal_cell,  agent.spec)

    # no camera_fn — we handle detection ourselves
    initial_path = agent.start_flight(start_gps, goal_gps, gps_fn=sim.gps)
    print(f"A* initial path: {len(initial_path)} cells from {start_cell} -> {goal_cell}")

    # camera schedule — step: image path
    n = len(initial_path)
    camera_schedule = {
        int(n * 0.28): "../testImages/trees2.jpg",
        int(n * 0.42): "../testImages/traffic-signals.png",
        int(n * 0.58): "../testImages/powerlines1.jpg",
        int(n * 0.75): "../testImages/trees2.jpg",
    }
    print(f"Camera triggers at steps {list(camera_schedule.keys())}")

    actual_path = [agent.cell]
    reroutes = 0
    detected_obstacles = []

    while not agent.at_goal:
        current_step = agent.step_count

        # check if camera fires at this step
        if current_step in camera_schedule:
            image_path = camera_schedule[current_step]
            print(f"\n[CAMERA] Step {current_step}: checking → {image_path}")

            if detect_obstacle(image_path):
                # block the next cell on the path
                
                if agent.path_index + 1 < len(agent.path):
                    blocked_cell = agent.path[agent.path_index + 1]
                    agent.grid[blocked_cell] = 1
                    agent.dstar.block_cell(blocked_cell)
                    new_path = agent.dstar.plan()
                    if new_path is None:
                        new_path = astar(agent.grid, agent.cell, agent.goal_cell)
                    if new_path is not None:
                        agent.path = new_path
                        agent.path_index = 0
                        reroute_steps.append(current_step)
                        reroutes += 1
                        detected_obstacles.append(blocked_cell)
                        print(f"  [step {current_step}] D* Lite reroute around {blocked_cell}")

        frame = agent.step()
        actual_path.append(agent.cell)
        path_snapshots.append(list(agent.path[agent.path_index:]))

        if frame.step > agent.grid.size:
            print("aborting: step limit reached")
            break

    print(f"\nGoal reached: {agent.at_goal}")
    print(f"Total steps : {agent.step_count}")
    print(f"Reroutes    : {reroutes}")
    print(f"Initial A*  : {len(initial_path)} cells")
    print(f"Actual path : {len(actual_path)} cells")

    if draw:
        visualize(agent, initial_path, actual_path, detected_obstacles,
              reroute_steps=reroute_steps, animate=True)



#  helpers
def nearest_free(grid: np.ndarray, cell: Tuple[int, int]) -> Tuple[int, int]:
    if grid[cell] == 0:
        return cell
    rows, cols = grid.shape
    for radius in range(1, max(rows, cols)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = cell[0] + dr, cell[1] + dc
                if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0:
                    return (r, c)
    return cell


if __name__ == "__main__":
    run_demo(use_osm=False, draw=True)