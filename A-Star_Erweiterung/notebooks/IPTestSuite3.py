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

obst = dict()
obst["obs1"] = LineString([(-2, 0), (-0.8, 0)]).buffer(0.5)
obst["obs2"] = LineString([(2, 0), (2, 1)]).buffer(0.2)
obst["obs3"] = LineString([(-1, 2), (1, 2)]).buffer(0.1)
description1 = "find a path around obstacles"
benchList.append(Benchmark("obst", KinChainCollisionChecker(r, obst, limits=limits), [[2.0, 0.5, 0.5]], [[-2.0, -0.5, -0.5]], description1, 2))

# -----------------------------------------

bottleNeckField = dict()
bottleNeckField["obs1"] = LineString([(-1, 3.5), (-4, 3.5)]).buffer(.25)
bottleNeckField["obs2"] = LineString([(1, 3.5), (4,3.5)]).buffer(.25)
description2 = "Planer has to find a narrow passage."
benchList.append(Benchmark("bottleNeck", KinChainCollisionChecker(r, bottleNeckField, limits=limits), [[2.25,0.5,0.5]], [[1.57,0,0]], description2, 2))