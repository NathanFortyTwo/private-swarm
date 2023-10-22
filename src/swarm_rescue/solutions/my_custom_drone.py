import math
import random
from typing import Optional
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
import numpy as np
import solutions.drone_utils as du

class MyCustomDrone(DroneAbstract):
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
        self.GPS_BOUNDS = (-500,500)
        self.MAP_SIZE = 10
        self.map = np.zeros([self.MAP_SIZE,self.MAP_SIZE])
        self.step_counter = 0
    
    def define_message_for_all(self):
        return self.map

    def process_lidar_sensor(self):
        pass

    def update_map(self):
        position = self.measured_gps_position()
        i,j = du.coords_to_indexes(position,self.GPS_BOUNDS,self.MAP_SIZE)
        if self.map[i,j] == 0:
            self.map[i,j] = 1
        
    def control(self):
        self.step_counter += 1
        self.update_map()
        print(self.map if self.step_counter%100==0 else "",end="\r")
        return {"forward": 0.7, "rotation": 0.2}