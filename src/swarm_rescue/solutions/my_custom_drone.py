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
        self.states=["EXPLORE"]
        self.grasper = 0
        self.arrived = False

    def define_message_for_all(self):
        return self.map

    def process_lidar_sensor(self):
        pass

    def update_map(self):
        position = self.measured_gps_position()
        i,j = du.coords_to_indexes(position,self.GPS_BOUNDS,self.MAP_SIZE)
        if self.map[i,j] == 0:
            self.map[i,j] = 1

    def get_base_path(self):
        coords = du.find_on_map(self.map,du.zone_types.BASE.value)
        if len(coords)==0:
            return None # c'est pas censé arriver
         
        position = self.measured_gps_position()
        i,j = du.coords_to_indexes(position,self.GPS_BOUNDS,self.MAP_SIZE)

         # une seule base donc on peut prendre le premier
        base = coords[0]
        path = du.astar(self.map,(i,j),base)
        self.path = path
        return self.path
    
    def get_closest_rescue_path(self):
        coords = du.find_on_map(self.map,du.zone_types.RESCUE.value)
        if len(coords)==0:
            return None
        
        position = self.measured_gps_position()
        i,j = du.coords_to_indexes(position,self.GPS_BOUNDS,self.MAP_SIZE)
        # find the path to the closest rescue
        length = math.inf
        for rescue in coords:
            path = du.astar(self.map,(i,j),rescue) # on peut changer pour prendre le premier path non null
            if path is not None:
                if len(path) < length:
                    self.path = path
        return self.path
        
    def move_to_next_point_in_path(self):
        # move to next point in path
        if len(self.path) <1:
            self.arrived= True
            return {"forward":0,"lateral":0,"rotation":0,"grasped":self.grasper}
        
        position = self.measured_gps_position()
        i,j = du.coords_to_indexes(position,self.GPS_BOUNDS,self.MAP_SIZE)
        next_point = self.path[0]
        if next_point[0] == i and next_point[1] == j: # si on est arrivé au point
            self.path.pop(0) # on enlève le point
            self.move_to_next_point_in_path() # on rappelle la fonction
        else: # sinon on se déplace en ligne droite
            return self.move_to(next_point[0],next_point[1])



    def move_to(self, i,j):
        # va a la prochaine case en ligne droite, normalement le chemin est sans obstacle
        #TODO

        return {"forward":0,"lateral":0,"grasper":0,"rotation":0}
    
    def control(self):
        self.step_counter += 1
        self.update_map()
        
        if "GOTO_RESCUE" in self.states :
            if self.path is None:
                self.path = self.get_closest_rescue_path()
            if self.path is not None:
                if self.arrived:
                    self.states.remove("GOTO_RESCUE")
                    self.states.append("GOTO_BASE")
                    self.path = None
                else:
                    return self.move_to_next_point_in_path() # appelé a chaque step de SAVE_RESCUE 
                

        if "GOTO_BASE" in self.states:
            if self.path is None:
                self.path = self.get_base_path()
            if self.path is not None:
                if self.arrived:
                    self.states.remove("GOTO_BASE")
                    self.states.append("EXPLORE")
                    self.path = None
                else:
                    return self.move_to_next_point_in_path()
            

        if "EXPLORE" in self.states:
            pass