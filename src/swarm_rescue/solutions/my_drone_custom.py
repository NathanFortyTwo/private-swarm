import numpy as np
import math
import random
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle
from spg_overlay.utils.pose import Pose


from solutions.utils.lidar_to_grid import OccupancyGrid



class MyDroneCustom(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False

        # Grid loop initialisation
        self.iteration: int = 0

        # Drone position
        self.estimated_pose = Pose()
        # Grid resolution (to determine)
        resolution = 10
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())


    def define_message_for_all(self):
        """
        We must transmit our gridmap to the other drones
        """
        pass

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided

    def control(self):
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit
        The Drone has also an occupancy grid to determine walls and empty spaces positions in the world
        """

        command_straight = {"forward": 0.7,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        command_turn = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.7,
                        "grasper": 0}

        # increment the iteration counter for the grid
        self.iteration += 1

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle())
        # self.estimated_pose = Pose(np.asarray(self.true_position()), self.true_angle()) -> only for debugging

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid, self.estimated_pose, title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid, self.estimated_pose, title="zoomed occupancy grid")
            # pass

        collided = self.process_lidar_sensor()

        self.counterStraight += 1

        if collided and not self.isTurning and self.counterStraight > self.distStopStraight:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        measured_angle = 0
        if self.measured_compass_angle() is not None:
            measured_angle = self.measured_compass_angle()

        diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
        if self.isTurning and abs(diff_angle) < 0.2:
            self.isTurning = False
            self.counterStraight = 0
            self.distStopStraight = random.uniform(10, 50)

        if self.isTurning:
            return command_turn
        else:
            return command_straight
