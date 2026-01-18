# coding: utf-8
"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).
Modified to support variable discretization, off-grid connections, edge collision checks, and optional reopening.
"""

import copy
import networkx as nx
import heapq
import math
from scipy.spatial.distance import euclidean, cityblock
from IPPlanerBase import PlanerBase
from IPPerfMonitor import IPPerfMonitor

class AStar(PlanerBase):
    def __init__(self, collChecker=0):
        """Constructor: Initialize all necessary members"""
        super(AStar, self).__init__(collChecker)
        self.graph = nx.DiGraph()  # = CloseList
        self.openList = []  # (<value>, <node>)

        self.goal = []
        self.goalFound = False

        self.limits = self._collisionChecker.getEnvironmentLimits()
        
        # Parameters set in planPath
        self.num_steps = None
        self.step_size = None
        self.w = 0.5  
        self.reopening = False
        return

    def _getNodeID(self, pos):
        """Compute a unique identifier based on the position"""
        nodeId = "-"
        for i in pos:
            # Round to avoid float precision problems
            nodeId += str(round(i, 4)) + "-"
        return nodeId
    
    def _getinGrid(self, pos):
        """Snap a real position to the nearest grid coordinates"""
        newpos = []
        for i, val in enumerate(pos):
            # Calculate nearest step index
            num_of_steps = round((val - self.limits[i][0]) / self.step_size[i])
            # Reconstruct position from index
            newval = self.limits[i][0] + num_of_steps * self.step_size[i]
            newpos.append(newval)
        return newpos

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """
        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information
        """
        # 0. reset
        self.graph.clear()
        self.openList = []      
        self.goalFound = False  
        self.solutionPath = []

        try:
            # 1. check start and goal whether collision free
            checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)

            # 2. Config reading
            self.w = config.get("w", 0.5)
            self.heuristic = config.get("heuristic", "euclidean")
            self.checkEdgeCollision = config.get("checkEdgeCollision", False)
            self.reopening = config.get("reopening", False) 

            # --- Discretization Setup ---
            num_dimensions = len(self.limits)
            if "num_steps" in config:
                if isinstance(config["num_steps"], list):
                    if len(config["num_steps"]) != num_dimensions:
                        raise ValueError(f"num_steps size mismatch")
                    self.num_steps = config["num_steps"]
                else:
                    self.num_steps = [config["num_steps"]] * num_dimensions
            else:
                self.num_steps = [44] * num_dimensions
            
            # Calculate step_size
            self.step_size = []
            for i, limit in enumerate(self.limits):
                self.step_size.append(round((limit[1] - limit[0]) / self.num_steps[i], 4))
            
            self.start = checkedStartList[0]
            self.goal = checkedGoalList[0]

            grid_start = self._getinGrid(self.start)
            grid_goal  = self._getinGrid(self.goal)
            grid_goalID = self._getNodeID(grid_goal)

            # Check connection: Real Start -> Grid Start
            if self._collisionChecker.lineInCollision(self.start, grid_start):
                print("Error: Cannot connect Start Position to the Grid!")
                return None
            
            # Check connection: Grid Goal -> Real Goal
            if self._collisionChecker.lineInCollision(grid_goal, self.goal):
                print("Error: Cannot connect Grid Goal to the Goal Position!")
                return None

            # --- START NODE LOGIC ---
            epsilon = 1e-3
            dist_start = euclidean(self.start, grid_start)

            if dist_start > epsilon:
                # Add Real Start manually
                RealstartID = self._getNodeID(self.start)
                self.graph.add_node(RealstartID, pos=self.start, status='closed', g=0)
                
                # Start A* at Grid Start, with Real Start as Parent
                self._addGraphNode(grid_start, RealstartID)
            else:
                # Start directly at Grid Start
                self._addGraphNode(grid_start)
            
            # --- MAIN LOOP ---
            currentBestName = self._getBestNodeName()
            breakNumber = 0
            max_iterations = 200000

            while currentBestName:
                if breakNumber > max_iterations:
                    print(f"A* Warning: Max iterations ({max_iterations}) reached.")
                    break
                
                breakNumber += 1
                currentBest = self.graph.nodes[currentBestName]

                # --- GOAL CHECK ---
                if currentBestName == grid_goalID:
                    dist_goal = euclidean(self.goal, grid_goal)
                    finalNodeName = currentBestName # Default: Grid Goal

                    if dist_goal > epsilon:
                        # Add Real Goal node
                        realGoalID = self._getNodeID(self.goal)
                        
                        # Add to graph with Grid Goal as Parent
                        g_final = currentBest["g"] + dist_goal
                        self.graph.add_node(realGoalID, pos=self.goal, status='closed', g=g_final)
                        self.graph.add_edge(realGoalID, currentBestName)
                        
                        finalNodeName = realGoalID

                    # Collect Path
                    self.solutionPath = []
                    self._collectPath(finalNodeName, self.solutionPath)
                    
                    # IMPORTANT: Reverse path to be Start -> Goal
                    self.solutionPath.reverse()
                    
                    self.goalFound = True
                    break

                # Close Node
                currentBest["status"] = 'closed'
                if self._collisionChecker.pointInCollision(currentBest["pos"]):
                    currentBest['collision'] = 1
                    currentBestName = self._getBestNodeName()
                    continue
                self.graph.nodes[currentBestName]['collision'] = 0

                # Expand
                self._handleNode(currentBestName)
                
                # Next
                try:
                    currentBestName = self._getBestNodeName()
                except IndexError:
                    break

            if self.goalFound:
                return self.solutionPath
            else:
                return None
        except Exception as e:
            print("Planning failed:", e)
            import traceback
            traceback.print_exc()
            return None

    def _insertNodeNameInOpenList(self, nodeName):
        """Put node in OpenList"""
        heapq.heappush(self.openList, (self._evaluateNode(nodeName), nodeName))

    @IPPerfMonitor
    def _addGraphNode(self, pos, fatherName=None):
        """Add a node to the graph."""
        node_id = self._getNodeID(pos)

        # Create node if it doesn't exist
        if node_id not in self.graph.nodes:
            self.graph.add_node(node_id, pos=pos, status='open', g=0)

        # Handle Parent connection
        if fatherName is not None:
            self.graph.add_edge(node_id, fatherName)
            
            father_pos = self.graph.nodes[fatherName]["pos"]
            dist = euclidean(father_pos, pos)
            
            # Update G cost
            self.graph.nodes[node_id]["g"] = self.graph.nodes[fatherName]["g"] + dist

        self._insertNodeNameInOpenList(node_id)

    def _setLimits(self, lowLimit, highLimit):
        """ Sets the limits of the investigated search space """
        self.limits = []
        for i in range(len(lowLimit)):
            self.limits.append([lowLimit[i], highLimit[i]])
        return
    
    def _getBestNodeName(self):
        """ Returns the name of best node in OpenList """
        while self.openList:
            _, name = heapq.heappop(self.openList)
            
            # Safety check: if node was removed from graph (rare)
            if name not in self.graph.nodes:
                continue

            node = self.graph.nodes[name]
            
            # Lazy Deletion: 
            # If we extract a node that is already closed, it means it's an old entry 
            # in the heap (from before a reopening or duplicate add). Ignore it.
            if node["status"] == 'closed':
                continue
                
            return name
        return None

    @IPPerfMonitor
    def _handleNode(self, nodeName):
        """Generates possible successor positions with REOPENING supported"""
        node = self.graph.nodes[nodeName]
        currentPos = node["pos"]
        currentG = node["g"]

        for i in range(len(currentPos)):
            for u in [-1, 1]:
                newPos = copy.copy(currentPos)
                newPos[i] += u * self.step_size[i]
                
                if not self._inLimits(newPos):
                    continue
                
                # Edge Collision Check
                if self.checkEdgeCollision:
                    if self._collisionChecker.lineInCollision(currentPos, newPos):
                        continue

                newNodeID = self._getNodeID(newPos)
                dist = euclidean(currentPos, newPos)
                tentative_g = currentG + dist

                # --- REOPENING LOGIC ---
                if newNodeID in self.graph.nodes:
                    if self.reopening: # Correction : ajout des deux points
                        existing_node = self.graph.nodes[newNodeID]
                        # If new path is shorter
                        if tentative_g < existing_node["g"] - 1e-6:
                            # Update G
                            existing_node["g"] = tentative_g
                            
                            # Update Parent (Remove old edge, add new one)
                            old_fathers = list(self.graph.successors(newNodeID))
                            if old_fathers:
                                self.graph.remove_edge(newNodeID, old_fathers[0])
                            self.graph.add_edge(newNodeID, nodeName)
                            
                            # Re-insert into OpenList
                            if existing_node["status"] == 'closed':
                                existing_node["status"] = 'open'
                            
                            self._insertNodeNameInOpenList(newNodeID)
                    
                    # If reopening is False OR path is not shorter, do nothing
                    continue

                # --- NEW NODE ---
                # Check Point Collision only for new nodes
                if self._collisionChecker.pointInCollision(newPos):
                    continue

                self._addGraphNode(newPos, nodeName)

    @IPPerfMonitor
    def _handleNode9(self, nodeName):
        """Generates possible successor positions (Diagonal)"""
        node = self.graph.nodes[nodeName]
        for i in range(len(node["pos"])):
            for j in range(len(node["pos"])): 
                for u in [-1, 1]:
                    for v in [-1, 0, 1]:
                        newPos = copy.copy(node["pos"])
                        newPos[i] += u * self.step_size[i]
                        newPos[j] += v * self.step_size[j]
                        
                        if not self._inLimits(newPos):
                            continue
                        
                        newNodeID = self._getNodeID(newPos)
                        if newNodeID in self.graph.nodes:
                            continue
                        
                        if self.checkEdgeCollision:
                            if self._collisionChecker.lineInCollision(node["pos"], newPos):
                                continue

                        self._addGraphNode(newPos, nodeName)

    @IPPerfMonitor
    def _computeHeuristicValue(self, nodeName):
        """ Computes Heuristic Value """
        node = self.graph.nodes[nodeName]
        if self.heuristic == "euclidean":
            return euclidean(self.goal, node["pos"])
        else:
            return cityblock(self.goal, node["pos"])

    @IPPerfMonitor
    def _evaluateNode(self, nodeName):
        node = self.graph.nodes[nodeName]
        return self.w * self._computeHeuristicValue(nodeName) + (1 - self.w) * node["g"]
                      
    def _collectPath(self, nodeName, solutionPath):
        # Recursively collects path from Child to Father
        solutionPath.append(nodeName)
        fathers = list(self.graph.successors(nodeName))
        if len(fathers) > 0:
            self._collectPath(fathers[0], solutionPath)
  
    @IPPerfMonitor
    def _inLimits(self, pos):
        for i, limit in enumerate(self.limits):
            # Added slight tolerance for float errors
            if pos[i] < limit[0] - 1e-5 or pos[i] > limit[1] + 1e-5:
                return False
        return True