import numpy as np
from enum import Enum
import numpy as np
import heapq

states_enum = Enum('States', ["INIT", "GOTO_TARGET", "EXPLORE"]) 
zone_types = Enum('ZoneTypes', ["UNKNOWN", "FREE", "WALL", "RESCUE", "BASE","NO_COMM","NO_GPS","DANGER"])
zone_cost = {zone_types.UNKNOWN.value:2,zone_types.FREE.value:1,zone_types.WALL.value:100,zone_types.RESCUE.value:1,zone_types.BASE.value:1,zone_types.NO_COMM.value:20,zone_types.NO_GPS.value:20,zone_types.DANGER.value:100}
def coords_to_indexes(coords, gpsbounds,matrix_size):
    """
    Converts coordinates to indexes in the matrix
    Matrix is a square matrix of size matrix_size
    """
    x,y = coords
    xmin, xmax = gpsbounds
    ymin,ymax = xmin,xmax
    n= matrix_size # nbline; nbcolumn
    i = (y-ymax)/(ymin-ymax)
    j = (x-xmin)/(xmax-xmin)
    return int(i*n),int(j*n)

def find_on_map(matrix, value):
    """
    finds the coordinates of the value in the matrix
    """
    positions =np.where(matrix==value)
    positions = list(zip(positions[0],positions[1]))
    return positions


# A* algorithm, c'est pas de  moi ça vient d'internet, 
# je l'ai adapté pour notre cas + testé 

# Define a Node class to represent a cell in the matrix
class Node:
    def __init__(self, x, y, cost, heuristic, parent):
        self.x = x
        self.y = y
        self.cost = cost  # g(n)
        self.heuristic = heuristic  # h(n)
        self.total_cost = cost + heuristic  # f(n) = g(n) + h(n)
        self.parent = parent

    def __lt__(self, other):
        return self.total_cost < other.total_cost

#distance manhattan, on peut changer et prendre la distance euclidienne
def heuristic(x1, y1, x2, y2):
    # return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return abs(x1 - x2) + abs(y1 - y2) 

# Check if the given coordinates are valid (inside matrix and not a wall)
def is_valid(matrix, x, y):
    return 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and (matrix[x][y] not in [zone_types.WALL.value]) 

def cost(matrix, x, y):
    value = int(matrix[x][y])
    return zone_cost[value]
    

def astar(matrix, start, goal):
    moves = [(0, -1), (1, 0), (0, 1), (-1, 0),]#(1, 1), (-1, -1), (-1, 1), (1, -1)] 
    # on peut enlever les diagonales si c'est trop dangereux pour les murs

    start_node = Node(start[0], start[1], 0, heuristic(start[0], start[1], goal[0], goal[1]), None)
    end_node = Node(goal[0], goal[1], 0, 0, None)

    open_list = [start_node]
    closed_list = set()

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.x == end_node.x and current_node.y == end_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_list.add((current_node.x, current_node.y))

        for move in moves:
            new_x, new_y = current_node.x + move[0], current_node.y + move[1]

            if not is_valid(matrix, new_x, new_y) or (new_x, new_y) in closed_list:
                continue

            new_cost = current_node.cost + cost(matrix, new_x, new_y)
            new_heuristic = heuristic(new_x, new_y, goal[0], goal[1])

            existing_node = next((n for n in open_list if n.x == new_x and n.y == new_y), None)
            if not existing_node or new_cost < existing_node.cost:
                if existing_node:
                    open_list.remove(existing_node)
                new_node = Node(new_x, new_y, new_cost, new_heuristic, current_node)
                heapq.heappush(open_list, new_node)

    return None # pas de chemin