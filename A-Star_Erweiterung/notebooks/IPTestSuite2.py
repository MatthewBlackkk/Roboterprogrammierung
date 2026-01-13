# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).
It gathers all visualizations of the investigated and explained planning algorithms.
License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPBenchmark import Benchmark 
from IPEnvironmentShapeRobot import CollisionCheckerShapeRobot
from IPEnvironmentShapeRobot import ShapeRobot, ShapeRobotWithOrientation
from shapely.geometry import Point, Polygon, LineString
import shapely.affinity
import math
import numpy as np


benchList = list()
robot_shape = Polygon([(-0.5, -0.5), (2, -0.5), (2.0, 0.5), (-0.5, 0.5)])
shape_robot = ShapeRobotWithOrientation(robot_shape, limits=[[0.0, 22.0,], [0.0, 22.0], [-math.pi, math.pi]])

# Neuer komplexer Roboter (z.B. L-förmig)
robot_shape_complex = Polygon([(0, 0), (2, 0), (2, 0.5), (0.5, 0.5), (0.5, 2), (0, 2)])
shape_robot_complex = ShapeRobotWithOrientation(robot_shape_complex, limits=[[0.0, 22.0,], [0.0, 22.0], [-math.pi, math.pi]])

# -----------------------------------------
trapField = dict()
trapField["obs1"] =   LineString([(6, 18), (6, 8), (16, 8), (16,18)]).buffer(1.0)
description = "Following the direct connection from goal to start would lead the algorithm into a trap."
benchList.append(Benchmark("Trap", CollisionCheckerShapeRobot(trapField, shape_robot), [[10,15,0]], [[20,1,math.pi/2]], description, 2))

# -----------------------------------------
bottleNeckField = dict()
bottleNeckField["obs1"] = LineString([(0, 13), (10.5, 13)]).buffer(.5)  # Ende bei 10 statt 11
bottleNeckField["obs2"] = LineString([(13.5, 13), (23,13)]).buffer(.5)  # Start bei 14 statt 13
description = "Planer has to find a narrow passage."
benchList.append(Benchmark("Bottleneck", CollisionCheckerShapeRobot(bottleNeckField, shape_robot), [[4,15,0]], [[20,10,math.pi/2]], description, 2))

# -----------------------------------------
fatBottleNeckField = dict()
fatBottleNeckField["obs1"] = Polygon([(0, 8), (11, 8),(11, 15), (0, 15)]).buffer(.5)
fatBottleNeckField["obs2"] = Polygon([(13, 8), (24, 8),(24, 15), (13, 15)]).buffer(.5)
description = "Planer has to find a narrow passage with a significant extend."
#benchList.append(Benchmark("Fat bottleneck", CollisionCheckerShapeRobot(fatBottleNeckField, shape_robot), [[4,21,0]], [[18,1,math.pi/2]], description, 2))

# -----------------------------------------

myField = dict()
myField["L"] = Polygon([(10, 16), (10, 11), (13, 11), (13,12), (11,12), (11,16)])
myField["T"] = Polygon([(14,16), (14, 15), (15, 15),(15,11), (16,11), (16,15), (17, 15), (17, 16)])
myField["C"] = Polygon([(19, 16), (19, 11), (22, 11), (22, 12), (20, 12), (20, 15), (22, 15), (22, 16)])

myField["Antenna_L"] = Polygon([(3, 12), (1, 16), (2, 16), (4, 12)])
myField["Antenna_Head_L"] = Point(1.5, 16).buffer(1)

myField["Antenna_R"] = Polygon([(7, 12), (9, 16), (8, 16), (6, 12)])
myField["Antenna_Head_R"] = Point(8.5, 16).buffer(1)

myField["Rob_Head"] = Polygon([(2, 13), (2, 8), (8, 8), (8, 13)])
description = "Planer has to find a passage past a robot head and the print of the LTC."
#benchList.append(Benchmark("MyField", CollisionCheckerShapeRobot(myField, shape_robot), [[4,21,0]], [[18,1,math.pi/2]], description, 2))

# Trap mit komplexem Roboter
trapField_complex = dict()
trapField_complex["obs1"] = LineString([(6, 18), (6, 8), (16, 8), (16,18)]).buffer(1.0)
description_complex = "Trap scenario with complex L-shaped robot."
benchList.append(Benchmark("Trap_Complex", CollisionCheckerShapeRobot(trapField_complex, shape_robot_complex), [[10,15,0]], [[20,1,math.pi/2]], description_complex, 2))

# Bottleneck mit komplexem Roboter
bottleNeckField_complex = dict()
bottleNeckField_complex["obs1"] = LineString([(0, 13), (10.5, 13)]).buffer(.5)
bottleNeckField_complex["obs2"] = LineString([(13.5, 13), (23,13)]).buffer(.5)
description_bottleneck = "Bottleneck with complex L-shaped robot."
benchList.append(Benchmark("Bottleneck_Complex", CollisionCheckerShapeRobot(bottleNeckField_complex, shape_robot_complex), [[4,15,0]], [[18,1,math.pi/2]], description_bottleneck, 2))
