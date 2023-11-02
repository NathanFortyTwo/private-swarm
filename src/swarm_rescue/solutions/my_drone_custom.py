import numpy as np
import math
import random
import sys
import os
from typing import Optional
from enum import Enum
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, rad2deg, deg2rad
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import circular_mean, clamp

from solutions.utils.lidar_to_grid import OccupancyGrid
from solutions.utils.Astar_pathfinder import AStarPlanner

show_animation = False


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
        self.isTurning = True
        self.pose_init = None
        self.epsilon_angle = 0.2

        # custom control
        self.collided_nums = 0

        # initial state
        self.state = self.Activity.EXPLORING

        # Grid loop initialisation
        self.iteration: int = 0

        # Drone position
        self.estimated_pose = Pose()
        self.counter_position = 0
        #self.counter_angle = 0

        # Grid resolution (to determine)
        resolution = 8
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())

        # path A*
        self.path = None
        self.current_angle = None
        self.limit_time_position_blocked = 70
        #self.limit_time_angle_blocked = 20
        self.transition = False

        # PD controller
        self.prev_diff_angle = 0
        self.prev_diff_position = 0

    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        EXPLORING = 1
        GRASPING_WOUNDED = 2
        GOING_TO_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

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

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.EXPLORING
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 1000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.GOING_TO_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 50)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.5, 1)

        return found_wounded, found_rescue_center, command

    def control(self):

        # counter increase if the drone is blocked
        # grid_estimated_position_x, grid_estimated_position_y = self.grid._conv_world_to_grid(self.estimated_pose.position[0], self.estimated_pose.position[1])
        # grid_true_position_x, grid_true_position_y = self.grid._conv_world_to_grid(self.true_position()[0], self.true_position()[1])
        # if self.grid._conv_world_to_grid(self.estimated_pose.position[0], self.estimated_pose.position[1])  != self.grid._conv_world_to_grid(self.true_position()[0], self.true_position()[1]):
        # if abs(grid_estimated_position_x - grid_true_position_x) < 2 and abs(grid_estimated_position_y - grid_true_position_y) < 2:
        #    self.counter_position += 1
        # if normalize_angle(self.estimated_pose.orientation - self.true_angle()) < self.epsilon_angle and self.counter_position > self.limit_time_blocked:
        #    self.counter_angle += 1
        # else:
        #   self.counter_position = 0
        #   self.counter_angle = 0

        #############
        # GRIDMAP + POSE UPDATE
        #############

        # increment the iteration counter for the grid
        self.iteration += 1


        # self.estimated_pose = Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle())
        self.estimated_pose = Pose(np.asarray(self.true_position()), self.true_angle())  # only for debugging

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0 and show_animation:
            self.grid.display(self.grid.grid, self.estimated_pose, title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid, self.estimated_pose, title="zoomed occupancy grid")
            # pass

        #############
        # SENSORS MEASURES AND COMMAND/POSE INIT
        #############

        command = {"forward": 0,
                   "lateral": 0,
                   "rotation": 0,
                   "grasper": 0}

        if self.pose_init is None:
            self.pose_init = self.measured_gps_position()

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.EXPLORING and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.Activity.GOING_TO_RESCUE_CENTER

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.Activity.EXPLORING

        elif self.state is self.Activity.GOING_TO_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = self.Activity.EXPLORING

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.GOING_TO_RESCUE_CENTER

        # print("state: {}, can_grasp: {}, grasped entities: {}".format(self.state.name,
        #                                                              self.base.grasper.can_grasp,
        #                                                             self.base.grasper.grasped_entities))

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a rescue center or wounded person is detected, we use a special command
        ##########
        if self.state is self.Activity.EXPLORING:
            command = self.control_custom()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.GOING_TO_RESCUE_CENTER:
            command = self.control_to_base()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command

    def control_custom(self):
        # reinitialize the path
        self.path = None

        command_straight = {"forward": 1,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        command_turn = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 1,
                        "grasper": 0}

        #############
        # PROCESS EXPLORING
        #############

        collided = self.process_lidar_sensor()
        self.counterStraight += 1

        if self.collided_nums == 0 or self.collided_nums == 4:

            if self.collided_nums == 0:
                self.isTurning = True
                self.angleStopTurning = - 7 * math.pi / 12

            if collided and not self.isTurning and self.counterStraight > 50:
                self.isTurning = True
                self.angleStopTurning = - 3 * math.pi / 4

            measured_angle = 0
            if self.measured_compass_angle() is not None:
                measured_angle = self.measured_compass_angle()

            diff_angle = normalize_angle(self.angleStopTurning - self.true_angle())
            if self.isTurning and abs(diff_angle) < self.epsilon_angle:
                self.isTurning = False
                self.collided_nums += 1
                self.counterStraight = 0

            if self.isTurning:
                return command_turn
            else:
                return command_straight

        if self.collided_nums == 1 or self.collided_nums == 3:

            if collided and not self.isTurning and self.counterStraight > 50:
                self.isTurning = True
                self.angleStopTurning = 9 * math.pi / 10

            measured_angle = 0
            if self.measured_compass_angle() is not None:
                measured_angle = self.measured_compass_angle()

            diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
            if self.isTurning and abs(diff_angle) < self.epsilon_angle:
                self.collided_nums += 1
                self.isTurning = False
                self.counterStraight = 0

            if self.isTurning:
                return command_turn
            else:
                return command_straight

        if self.collided_nums == 2:

            if collided and not self.isTurning and self.counterStraight > 50:
                self.isTurning = True
                self.angleStopTurning = math.pi / 4

            measured_angle = 0
            if self.measured_compass_angle() is not None:
                measured_angle = self.measured_compass_angle()

            diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
            if self.isTurning and abs(diff_angle) < self.epsilon_angle:
                self.collided_nums += 1
                self.isTurning = False
                self.counterStraight = 0

            if self.isTurning:
                return command_turn
            else:
                return command_straight

        if self.collided_nums == 5:

            if collided and not self.isTurning and self.counterStraight > 50:
                self.isTurning = True
                self.angleStopTurning = - 2 * math.pi / 5

            measured_angle = 0
            if self.measured_compass_angle() is not None:
                measured_angle = self.measured_compass_angle()

            diff_angle = normalize_angle(self.angleStopTurning - measured_angle)
            if self.isTurning and abs(diff_angle) < self.epsilon_angle:
                self.isTurning = False
                self.collided_nums += 1
                self.counterStraight = 0

            if self.isTurning:
                return command_turn
            else:
                return command_straight

        else:
            return command_straight

    def control_exploring(self):
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

        #############
        # PROCESS EXPLORING
        #############

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

    def control_to_base(self):

        command_base = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}

        if self.path is None:

            ox, oy = self.grid.get_index_obstacles

            sx, sy = self.grid._conv_world_to_grid(self.estimated_pose.position[0], self.estimated_pose.position[1])
            gx, gy = self.grid._conv_world_to_grid(self.pose_init[0], self.pose_init[1])

            # maj from grid to A*
            sx, sy = self.grid.Astar_to_grid_index(sx, sy)
            gx, gy = self.grid.Astar_to_grid_index(gx, gy)

            if show_animation:  # pragma: no cover
                plt.plot(ox, oy, ".k")
                plt.plot(sx, sy, "og")
                plt.plot(gx, gy, "xb")
                plt.grid(True)
                plt.axis("equal")

            planner = AStarPlanner(ox, oy, self.grid.get_resolution, show_animation)
            rx, ry = planner.planning(sx, sy, gx, gy)

            if show_animation:  # pragma: no cover
                plt.plot(rx, ry, "-r")
                plt.pause(0.3)

            # from float to int coordinates
            int_rx = [round(x) for x in rx]
            int_ry = [round(y) for y in ry]

            # maj from A* to grid
            target_x_path = []
            target_y_path = []
            for x, y in zip(int_rx, int_ry):
                x, y = self.grid.Astar_to_grid_index(x, y)
                target_x_path.append(x)
                target_y_path.append(y)

            # correct order of the path
            target_x_path = target_x_path[::-1]
            target_y_path = target_y_path[::-1]

            # slicing the path for a better performance
            new_target_x_path = target_x_path[0:len(target_x_path):3]
            new_target_y_path = target_y_path[0:len(target_y_path):3]

            self.path = [new_target_x_path, new_target_y_path]
            self.current_angle = self.estimated_pose.orientation

        # print(rad2deg(self.current_angle))
        # Come back to base
        command = self.move_to_next_point_in_path(self.path[0], self.path[1], command_base,
                                                  self.estimated_pose.position, self.estimated_pose.orientation)
        return command

    def move_to_next_point_in_path(self, x_path, y_path, command, position, angle):
        i, j = self.grid._conv_world_to_grid(position[0], position[1])
        next_x, next_y = x_path[0], y_path[0]
        if next_x == i and next_y == j:  # si on est arrivé au point
            # on enlève le point
            del x_path[0]
            del y_path[0]
            self.counter_position = 0
            #self.counter_angle = 0
            return self.move_to_next_point_in_path(x_path, y_path, command, position, angle)  # on rappelle la fonction
        else:  # sinon on se déplace en ligne droite
            """
            if not self.transition:
                self.counter_position += 1
            if self.counter_position == self.limit_time_position_blocked:
                self.current_angle = self.estimated_pose.orientation
                # print(rad2deg(self.current_angle))
                self.counter_position += 1
            """
            return self.move_to(next_x, next_y, command, position, angle)

    def move_to(self, target_i, target_j, command, position, angle):
        # va a la prochaine case en ligne droite, normalement le chemin est sans obstacle
        i, j = self.grid._conv_world_to_grid(position[0], position[1])
        if self.iteration % 5 == 0:
            print(
                f"counter_position={self.counter_position}, angle={rad2deg(angle)}, current_angle={rad2deg(normalize_angle(self.current_angle))}")


        if self.counter_position >= self.limit_time_position_blocked:

            self.transition = True
            #if self.counter_angle <= self.limit_time_angle_blocked:
            if self.adjust_compass(angle, normalize_angle(self.current_angle + math.pi / 2), command) is not None:
                #self.counter_angle += 1
                command = self.adjust_compass(angle, normalize_angle(self.current_angle + math.pi / 2), command)
                # print(command)
                return command
            else:
                self.current_angle = normalize_angle(self.current_angle + math.pi / 2)
                self.counter_position = 0
                #self.counter_angle = 0
                self.transition = False
                print("transition terminée!")
                return self.move_to(target_i, target_j, command, position, angle)
            """
            else:
                if self.adjust_compass(angle, self.current_angle - math.pi / 2, command) is not None:
                    command = self.adjust_compass(angle, self.current_angle - math.pi / 2, command)
                    # print(command)
                    return command
                else:
                    self.current_angle = self.current_angle - math.pi / 2
                   self.counter_position = 0
                    self.counter_angle = 0
                    self.transition = False
                    return self.move_to(target_i, target_j, command, position, angle)
            """
        else:
            #print("je vais à la target")
            if i < target_i and j < target_j:
                command["forward"] = MyDroneCustom.command_proportional(math.cos(angle) - math.sin(angle))
                command["lateral"] = MyDroneCustom.command_proportional(-math.sin(angle) - math.cos(angle))


            elif i < target_i and j > target_j:
                command["forward"] = MyDroneCustom.command_proportional(math.cos(angle) + math.sin(angle))
                command["lateral"] = MyDroneCustom.command_proportional(-math.sin(angle) + math.cos(angle))


            elif i > target_i and j > target_j:
                command["forward"] = MyDroneCustom.command_proportional(-math.cos(angle) + math.sin(angle))
                command["lateral"] = MyDroneCustom.command_proportional(math.sin(angle) + math.cos(angle))


            elif i > target_i and j < target_j:
                command["forward"] = MyDroneCustom.command_proportional(-math.cos(angle) - math.sin(angle))
                command["lateral"] = MyDroneCustom.command_proportional(math.sin(angle) - math.cos(angle))


            elif i == target_i and j < target_j:
                command["forward"] = -math.sin(angle)
                command["lateral"] = -math.cos(angle)


            elif i == target_i and j > target_j:
                command["forward"] = math.sin(angle)
                command["lateral"] = math.cos(angle)


            elif i < target_i and j == target_j:
                command["forward"] = math.cos(angle)
                command["lateral"] = -math.sin(angle)


            elif i > target_i and j == target_j:
                command["forward"] = -math.cos(angle)
                command["lateral"] = math.sin(angle)

        """
            if j < target_j:
                command["forward"] = -math.sin(angle)
                command["lateral"] = -math.cos(angle)

            elif j > target_j:
                command["forward"] = math.sin(angle)
                command["lateral"] = math.cos(angle)

            elif i < target_i :
                command["forward"] = math.cos(angle)
                command["lateral"] = -math.sin(angle)

            elif i > target_i:
                command["forward"] = -math.cos(angle)
                command["lateral"] = math.sin(angle)
            """

        command = self.coeff_value_command(target_i, target_j, i, j, command)
        return command



    """
    def adjust_angle(self, angle, command):
        
        if self.counter_angle <= self.limit_time_angle_blocked:
            diff_angle_1 = normalize_angle(self.current_angle + math.pi/2 - angle)
            if abs(diff_angle_1) < self.epsilon_angle:
                self.current_angle = self.current_angle + math.pi/2
                return None
            elif diff_angle_1 > 0:
                command["rotation"] = 0.4
                return command
            else:
                command["rotation"] = -0.4
                return command

        else:
            diff_angle_2 = normalize_angle(self.current_angle - math.pi/2 - angle)
            if abs(diff_angle_2) < self.epsilon_angle:
                self.current_angle = self.current_angle - math.pi / 2
                return None
            elif diff_angle_2 > 0:
                command["rotation"] = 0.7
                return command
            else:
                command["rotation"] = -0.7
                return command
        """

    @staticmethod
    def command_proportional(value):
        return (math.sqrt(2) / 2) * value

    def coeff_value_command(self, target_x, target_y, x_position, y_position, command):
        diff_position = math.sqrt((target_x - x_position)**2 + (target_y - y_position)**2)
        deriv_diff_position = diff_position - self.prev_diff_position

        # PD filter 1 parameters #########
        Ku = 25 / 100  # Gain debut oscillation maintenue en P pure
        Tu = 26  # Période d'oscillation
        Kp = 0.8 * Ku
        Kd = Ku * Tu / 10.0

        # PD controller equation
        coeff = Kp * diff_position + Kd * deriv_diff_position
        coeff = abs(clamp(coeff, -1.0, 1.0))
        # for the moment no path tracker so fixed coeff
        coeff = 0.9
        self.prev_diff_position = diff_position

        for key in command:
            command[key] = coeff * command[key]
        return command

    def adjust_compass(self, angle, target_angle, command):
        diff_angle = normalize_angle(target_angle - angle)

        # PD controller parameters
        Ku = 11.16  # Gain debut oscillation maintenue en P pure
        Tu = 2.0  # Période d'oscillation
        Kp = 0.8 * Ku
        Kd = Ku * Tu / 40.0
        deriv_diff_angle = normalize_angle(diff_angle - self.prev_diff_angle)

        # PD controller equation
        rotation = Kp * diff_angle + Kd * deriv_diff_angle
        rotation = clamp(rotation, -1, 1)
        self.prev_diff_angle = diff_angle

        if abs(diff_angle) < self.epsilon_angle:
            return None
        else:
            command["rotation"] = rotation
            return command

