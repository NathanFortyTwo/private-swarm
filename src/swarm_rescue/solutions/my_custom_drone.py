import math
import random
from typing import Optional
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
import numpy as np
import solutions.drone_utils as du
from spg_overlay.utils import utils as u
import time
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
"""
Les méthodes qui ne sont PAS appellée par control() sont préfacées par un _
"""
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

        self.isTurning = False
        self.GPS_BOUNDS = (-500,500) # TODO trouver un moyen d'avoir ces valeurs
        self.resolution = 50
        self.MAP_SIZE = int(misc_data.size_area[0]/50) # on suppose que c'est un carré
        #TODO supposition a tej des que possible
        self.map = np.zeros([self.MAP_SIZE,self.MAP_SIZE])
        self.map.fill(du.zone_types.UNKNOWN.value)
        self.step_counter = 0
        self.states=["EXPLORE"]
        self.grasper = 0
        self.arrived = False
        self.path = None
        self.epsilon_angle = 5
        self.DIST_SEUIL_LIDAR = 150
        
        self.x = None
        self.y = None
        self.i = None
        self.j = None
        self.theta = None
        self.target = None

        print("MAP_SIZE : ",self.MAP_SIZE)
        print("GPS_BOUNDS : ",self.GPS_BOUNDS)
        print("number_drones : ",misc_data.number_drones)
        print("size_area : ",misc_data.size_area)
        

    def define_message_for_all(self):
        return {"map":self.map}

    def _process_lidar_sensor(self):

        res = []
        for (i,dist) in enumerate(self.lidar_values()):
            if dist < self.DIST_SEUIL_LIDAR:
                #sachant que lidar[0] est en bas du drone, et lidar[45] est a droite du drone
                angle = self.theta - math.pi + i*math.pi*2/180 # un peu de trigo
                xpos = self.x + dist*math.cos(angle)
                ypos = self.y + dist*math.sin(angle)
                res.append((xpos,ypos))
        return res

    def update_map_with_lidar(self):
        coords = self._process_lidar_sensor()
        for coord in coords:
            i,j = du.coords_to_indexes(coord,self.GPS_BOUNDS,self.MAP_SIZE)
            if self.map[i,j] == du.zone_types.UNKNOWN.value:
                self.map[i,j] = du.zone_types.WALL.value

    def update_map(self):
        i,j = self.i,self.j
        if self.map[i,j] == du.zone_types.UNKNOWN.value:
            self.map[i,j] = du.zone_types.FREE.value
        
        self.update_map_with_lidar()

    def get_base_path(self):
        coords = du.find_on_map(self.map,du.zone_types.BASE.value)
        if len(coords)==0:
            return None # c'est pas censé arriver
         
        i,j = self.i,self.j

         # une seule base donc on peut prendre le premier
        base = coords[0]
        path = du.astar(self.map,(i,j),base)
        self.path = path
        return self.path
    
    def get_closest_rescue_path(self):
        coords = du.find_on_map(self.map,du.zone_types.RESCUE.value)
        if len(coords)==0:
            return None
        
        i,j = self.i,self.j
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
        
        i,j = self.i,self.j
        next_point = self.path[0]
        if next_point[0] == i and next_point[1] == j: # si on est arrivé au point
            self.path.pop(0) # on enlève le point
            self.move_to_next_point_in_path() # on rappelle la fonction
        else: # sinon on se déplace en ligne droite
            return self.move_to(next_point[0],next_point[1])



    def move_to(self, target_i,target_j):
        # va a la prochaine case en ligne droite, normalement le chemin est sans obstacle
        i,j = self.i,self.j
        #print("je suis en",i,j,"je vais en",target_i,target_j)
        if i<target_i:
            return {"forward":-1,"lateral":0,"rotation":0,"grasper":self.grasper}
        elif target_i < i:
            return {"forward":1,"lateral":0,"rotation":0,"grasper":self.grasper}
        elif j < target_j:
            return {"forward":0,"lateral":-1,"rotation":0,"grasper":self.grasper}
        else:
            return {"forward":0,"lateral":1,"rotation":0,"grasper":self.grasper}
    
    def get_random_path(self):
        assert self.path is None

        i,j = self.i,self.j
        targets = np.where(self.map == du.zone_types.UNKNOWN.value)
        if len(targets[0])==0:
            return None
        
        while self.path is None:
            target_i,target_j = random.choice(list(zip(targets[0],targets[1])))
            print("trying target ",target_i,target_j)

            self.path = du.astar(self.map,(i,j),(target_i,target_j))
            self.target = (target_i,target_j)

    def adjust_compass(self):
        angle = u.rad2deg(self.theta)
        target_angle = u.rad2deg(math.pi/2)
        if abs(angle-target_angle) <self.epsilon_angle:
            return None
        if angle < target_angle:
            return {"forward":0,"lateral":0,"rotation":1,"grasper":self.grasper}
        else:
            return {"forward":0,"lateral":0,"rotation":-1,"grasper":self.grasper}


    def check_still_valid_path(self):
        # vérifie que le chemin est toujours valide
        if self.path is None:
            return True
        for point in self.path:
            i,j = point
            if self.map[i,j] == du.zone_types.WALL.value:
                return False
        return True

    def process_semantic(self):
        print(self.semantic_values())


    def control(self):
        self.step_counter += 1
        self.x, self.y = self.measured_gps_position()
        self.i, self.j = du.coords_to_indexes((self.x,self.y),self.GPS_BOUNDS,self.MAP_SIZE)
        self.theta = self.measured_compass_angle()

        adjust_compass = self.adjust_compass()
        if adjust_compass is not None:
            return adjust_compass
        # apply kalman filter here if needed
    
        self.update_map()

        if not self.check_still_valid_path():
            print("path not valid anymore")
            self.path = du.astar(self.map,(self.i,self.j),self.target)

        if self.step_counter % 1 == 0:
            #print(self.map)
            print(self.semantic_values())
            #print("i'm at",self.i,self.j)
            #print("my path is",self.path)
            #print(np.where(self.map == du.zone_types.WALL.value))
        
        return
        if "GOTO_RESCUE" in self.states :
            if self.path is None:
                self.get_closest_rescue_path()
            if self.path is not None:
                if self.arrived:
                    self.grasper = 1
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
            if self.path is None:
                self.get_random_path()
                self.arrived = False
            if self.path is not None:
                if self.arrived:
                    self.path = None
                else:
                    return self.move_to_next_point_in_path()
