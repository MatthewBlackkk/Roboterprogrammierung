# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).
It gathers all visualizations of the investigated and explained planning algorithms.
License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPBenchmark import Benchmark 
from IPPlanarManipulator import PlanarJoint, PlanarRobot
from IPEnvironmentKin import KinChainCollisionChecker
from shapely.geometry import Point, Polygon, LineString
import shapely.affinity
import math
import numpy as np


benchList = list()
r = PlanarRobot(n_joints=3)
limits = [[-3.14,3.14],[-3.14,3.14],[-3.14,3.14]]

# -----------------------------------------
trapField = dict()
trapField["obs1"] =   LineString([(6, 18), (6, 8), (16, 8), (16,18)]).buffer(1.0)
description = "Following the direct connection from goal to start would lead the algorithm into a trap."
#benchList.append(Benchmark("Trap", KinChainCollisionChecker(r, trapField, limits=limits), [[10,15,0.5]], [[10,1,-0.5]], description, 2))

# -----------------------------------------
bottleNeckField = dict()
bottleNeckField["obs1"] = LineString([(0, 13), (11, 13)]).buffer(.5)
bottleNeckField["obs2"] = LineString([(13, 13), (23,13)]).buffer(.5)
description = "Planer has to find a narrow passage."
#benchList.append(Benchmark("Bottleneck", KinChainCollisionChecker(r, bottleNeckField, limits=limits), [[4,15,0]], [[18,1,math.pi/2]], description, 2))

# -----------------------------------------
fatBottleNeckField = dict()
fatBottleNeckField["obs1"] = Polygon([(0, 8), (11, 8),(11, 15), (0, 15)]).buffer(.5)
fatBottleNeckField["obs2"] = Polygon([(13, 8), (24, 8),(24, 15), (13, 15)]).buffer(.5)
description = "Planer has to find a narrow passage with a significant extend."
#benchList.append(Benchmark("Fat bottleneck", KinChainCollisionChecker(r, fatBottleNeckField, limits=limits), [[4,21,0]], [[18,1,math.pi/2]], description, 2))

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
#benchList.append(Benchmark("MyField", KinChainCollisionChecker(r, myField, limits=limits), [[4,21,0]], [[18,1,math.pi/2]], description, 2))

# -----------------------------------------

obst = dict()
obst["obs1"] = LineString([(-2, 0), (-0.8, 0)]).buffer(0.5)
obst["obs2"] = LineString([(2, 0), (2, 1)]).buffer(0.2)
obst["obs3"] = LineString([(-1, 2), (1, 2)]).buffer(0.1)
benchList.append(Benchmark("obst", KinChainCollisionChecker(r, obst, limits=limits), [[2.0, 0.5, 0.5]], [[-2.0, -0.5, -0.5]], description, 2))
