import os
import sys

# This line add, to sys.path, the path to parent path of this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spg_overlay.entities.normal_wall import NormalWall, NormalBox


# Dimension of the map : (700, 500)
# Dimension factor : 1.0
"""
def add_boxes(playground):
    # box 0
    box = NormalBox(up_left_point=(-556, -222),
                    width=433, height=153)
    playground.add(box, box.wall_coordinates)

    # box 1
    box = NormalBox(up_left_point=(6, 69),
                    width=130, height=100)
    playground.add(box, box.wall_coordinates)

    # box 2
    box = NormalBox(up_left_point=(-556, 375),
                    width=434, height=155)
    playground.add(box, box.wall_coordinates)
"""

def add_walls(playground):
    # vertical wall 0
    wall = NormalWall(pos_start=(-128, 0),
                      pos_end=(-128, 250))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 1
    wall = NormalWall(pos_start=(350, 180),
                      pos_end=(100, 180))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 2
    wall = NormalWall(pos_start=(10, 100),
                      pos_end=(10, -50))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 3
    wall = NormalWall(pos_start=(-350, -100),
                      pos_end=(-10, -100))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 4
    wall = NormalWall(pos_start=(170, 0),
                      pos_end=(170, -250))
    playground.add(wall, wall.wall_coordinates)

    """
    # vertical wall 5
    wall = NormalWall(pos_start=(148, 361),
                      pos_end=(148, 235))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 6
    wall = NormalWall(pos_start=(150, 361),
                      pos_end=(150, 235))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 7
    wall = NormalWall(pos_start=(336, 361),
                      pos_end=(336, 300))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 8
    wall = NormalWall(pos_start=(338, 361),
                      pos_end=(338, 300))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 9
    wall = NormalWall(pos_start=(-543, 227),
                      pos_end=(-543, -226))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 10
    wall = NormalWall(pos_start=(-542, 226),
                      pos_end=(-398, 226))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 11
    wall = NormalWall(pos_start=(-400, 226),
                      pos_end=(-400, -35))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 12
    wall = NormalWall(pos_start=(150, 161),
                      pos_end=(150, 137))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 13
    wall = NormalWall(pos_start=(150, 139),
                      pos_end=(338, 139))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 14
    wall = NormalWall(pos_start=(148, 160),
                      pos_end=(148, 136))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 15
    wall = NormalWall(pos_start=(148, 137),
                      pos_end=(338, 137))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 16
    wall = NormalWall(pos_start=(338, 139),
                      pos_end=(338, -109))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 17
    wall = NormalWall(pos_start=(-302, 138),
                      pos_end=(-302, 98))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 18
    wall = NormalWall(pos_start=(-302, 136),
                      pos_end=(-128, 136))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 19
    wall = NormalWall(pos_start=(336, 138),
                      pos_end=(336, -109))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 20
    wall = NormalWall(pos_start=(-300, 136),
                      pos_end=(-300, 98))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 21
    wall = NormalWall(pos_start=(-302, 134),
                      pos_end=(-128, 134))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 22
    wall = NormalWall(pos_start=(-130, 135),
                      pos_end=(-130, -226))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 23
    wall = NormalWall(pos_start=(-544, 114),
                      pos_end=(-526, 114))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 24
    wall = NormalWall(pos_start=(-436, 114),
                      pos_end=(-400, 114))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 25
    wall = NormalWall(pos_start=(-544, 112),
                      pos_end=(-526, 112))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 26
    wall = NormalWall(pos_start=(-436, 112),
                      pos_end=(-400, 112))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 27
    wall = NormalWall(pos_start=(9, 68),
                      pos_end=(9, -27))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 28
    wall = NormalWall(pos_start=(10, 67),
                      pos_end=(133, 67))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 29
    wall = NormalWall(pos_start=(130, 67),
                      pos_end=(130, -27))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 30
    wall = NormalWall(pos_start=(-300, 12),
                      pos_end=(-300, -226))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 31
    wall = NormalWall(pos_start=(-302, 11),
                      pos_end=(-302, -226))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 32
    wall = NormalWall(pos_start=(8, -25),
                      pos_end=(132, -25))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 33
    wall = NormalWall(pos_start=(-130, -107),
                      pos_end=(198, -107))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 34
    wall = NormalWall(pos_start=(278, -107),
                      pos_end=(336, -107))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 35
    wall = NormalWall(pos_start=(36, -107),
                      pos_end=(36, -277))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 36
    wall = NormalWall(pos_start=(-130, -109),
                      pos_end=(198, -109))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 37
    wall = NormalWall(pos_start=(276, -109),
                      pos_end=(338, -109))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 38
    wall = NormalWall(pos_start=(34, -108),
                      pos_end=(34, -277))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 39
    wall = NormalWall(pos_start=(-399, -116),
                      pos_end=(-399, -154))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 40
    wall = NormalWall(pos_start=(-400, -117),
                      pos_end=(-400, -154))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 41
    wall = NormalWall(pos_start=(-544, -142),
                      pos_end=(-400, -142))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 42
    wall = NormalWall(pos_start=(-544, -144),
                      pos_end=(-400, -144))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 43
    wall = NormalWall(pos_start=(-544, -224),
                      pos_end=(-130, -224))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 44
    wall = NormalWall(pos_start=(-129, 136),
                      pos_end=(-129, -364))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 45
    wall = NormalWall(pos_start=(-130, -362),
                      pos_end=(548, -362))
    playground.add(wall, wall.wall_coordinates)
    """

