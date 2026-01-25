# python
import copy
import heapq
import networkx as nx
from IPAStar import AStar
from scipy.spatial.distance import euclidean
from IPPerfMonitor import IPPerfMonitor


class ReopenAStar(AStar):
    def __init__(self, collChecker=0):
        """Contructor:

                Initialize all necessary members"""
        super(ReopenAStar, self).__init__(collChecker)

        self.graph = nx.DiGraph()  # = CloseList
        self.openList = []  # (<value>, <node>)

        self.start = []
        self.goal = []
        self.goalFound = False

        self.limits = self._collisionChecker.getEnvironmentLimits()

        self.num_steps = None
        self.step_size = None
        self.allowReopening = False

        self.w = 0.5
        return
    
    def _getBestNodeName(self):
        """Returns the best node name, skipping stale entries when reopening is active"""
        if not self.openList:
            return None

        # When reopening is disabled, use fast path
        if not self.allowReopening:
            if self.openList:
                return heapq.heappop(self.openList)[1]
            return None

        # When reopening is active, skip stale entries (nodes that are no longer 'open')
        while self.openList:
            _, nodeName = heapq.heappop(self.openList)
            node = self.graph.nodes.get(nodeName)
            if node is None:
                continue
            if node.get("status") == "open":
                return nodeName
        return None

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """

        Args:
            startList (list): list of start positions in planning space
            goalList (list): list of goal positions in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:

            config["w"] = 0.5
            config["heuristic"] = "euclidean"
            config["allowReopening"] = True

        """
        # 0. reset
        self.graph.clear()
        self.openList = []
        self.goalFound = False
        self.solutionPath = []

        try:
            # 1. check start and goal whether collision free (s. BaseClass)
            checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)

            # 2.
            self.w = config["w"]
            self.heuristic = config["heuristic"]
            self.allowReopening = config.get("allowReopening", False)

            num_dimensions = len(self.limits)
            if "num_steps" in config:
                if isinstance(config["num_steps"], list):
                    if len(config["num_steps"]) != num_dimensions:
                        raise ValueError(f"num_steps hat {len(config['num_steps'])} Elemente, aber Raum hat {num_dimensions} Dimensionen")
                    self.num_steps = config["num_steps"]
                else:
                    # Einzelner Wert für alle Dimensionen
                    self.num_steps = [config["num_steps"]] * num_dimensions
            else:
                # Default: 44 für alle Dimensionen
                self.num_steps = [44] * num_dimensions

            self.step_size = []
            for i, limit in enumerate(self.limits):
                self.step_size.append(round((limit[1] - limit[0]) / self.num_steps[i], 4))

            # Erweiterung für Kantenkollision -- Ludwig
            self.checkEdgeCollision = config.get("checkEdgeCollision", False)
            # Ende Erweiterung für Kantenkollision -- Ludwig

            self.start = checkedStartList[0]
            self.goal = checkedGoalList[0]

            grid_start = self._getinGrid(self.start)
            grid_goal = self._getinGrid(self.goal)
            grid_goalID = self._getNodeID(grid_goal)

            # Check connection: Real Start -> Grid Start
            if self._collisionChecker.lineInCollision(self.start, grid_start):
                print("Error: Cannot connect Start Position to the Grid!")
                return None

            # Check connection: Grid Goal -> Real Goal
            if self._collisionChecker.lineInCollision(grid_goal, self.goal):
                print("Error: Cannot connect Grid Goal to the Goal Position!")
                return None

            dist_start = euclidean(self.start, grid_start)
            epsilon = 1e-3

            if dist_start > epsilon:
                # CAS 1 : We are away from the grid
                # Add the real Start to the graph MANUALLY (without putting it in openList)
                RealstartID = self._getNodeID(self.start)
                self.graph.add_node(RealstartID, pos=self.start, status='closed', g=0)

                # Start A* on the grid point, with the real Start as PARENT
                self._addGraphNode(grid_start, RealstartID)
            else:
                # CAS 2 : We are already on the grid (or very close)
                self._addGraphNode(grid_start)

            currentBestName = self._getBestNodeName()
            breakNumber = 0

            max_iterations = 100000
            while currentBestName:
                if breakNumber > max_iterations:
                    print(f"A* Warnung: Max. Iterationen ({max_iterations}) erreicht. Graph hat {self.graph.number_of_nodes()} Knoten.")
                    break

                breakNumber += 1

                currentBest = self.graph.nodes[currentBestName]

                # Check if we reached the grid goal
                if currentBestName == grid_goalID:
                    dist_goal = euclidean(self.goal, grid_goal)

                    finalNodeName = currentBestName  # Default: grid point

                    if dist_goal > epsilon:
                        # If real Goal is far away, add it to graph now
                        realGoalID = self._getNodeID(self.goal)

                        # Add it with grid point as PARENT
                        g_final = currentBest["g"] + dist_goal
                        self.graph.add_node(realGoalID, pos=self.goal, status='closed', g=g_final)
                        self.graph.add_edge(realGoalID, currentBestName)

                        finalNodeName = realGoalID

                    self.solutionPath = []
                    self._collectPath(finalNodeName, self.solutionPath)
                    self.goalFound = True
                    break

                currentBest["status"] = 'closed'
                if self._collisionChecker.pointInCollision(currentBest["pos"]):
                    currentBest['collision'] = 1
                    currentBestName = self._getBestNodeName()
                    continue
                self.graph.nodes[currentBestName]['collision'] = 0

                # handleNode merges with former expandNode
                self._handleNode(currentBestName)
                currentBestName = self._getBestNodeName()

            if self.goalFound:
                return self.solutionPath
            else:
                return None
        except Exception as e:
            print(f"Planning failed: {e}")
            return None

    @IPPerfMonitor
    def _handleNode(self, nodeName):
        """Generates possible successor positions in all dimensions"""
        node = self.graph.nodes[nodeName]
        for i in range(len(node["pos"])):
            for u in [-1, 1]:
                newPos = copy.copy(node["pos"])
                newPos[i] += u * self.step_size[i]

                # Check if position is within limits
                if not self._inLimits(newPos):
                    continue

                # Edge collision check
                if self.checkEdgeCollision:
                    if self._collisionChecker.lineInCollision(node["pos"], newPos):
                        continue

                newNodeID = self._getNodeID(newPos)
                newG = node["g"] + euclidean(node["pos"], newPos)

                # Case 1: Node doesn't exist yet - create it
                if newNodeID not in self.graph:
                    self._addGraphNode(newPos, nodeName)
                    continue

                # Case 2: Node exists - check if we should update it (reopening)
                existingNode = self.graph.nodes[newNodeID]

                # If reopening is disabled and node exists, skip it (standard A* behavior)
                if not self.allowReopening:
                    continue

                # If reopening is enabled, check if we found a better path
                if newG < existingNode["g"]:
                    # Update the node with better cost
                    self.graph.nodes[newNodeID]["g"] = newG

                    # Remove old parent edge(s) and add new one
                    # Graph structure: child -> parent (stored as successor)
                    oldParents = list(self.graph.successors(newNodeID))
                    if oldParents:
                        self.graph.remove_edges_from([(newNodeID, p) for p in oldParents])

                    # Add new parent edge: child -> parent
                    self.graph.add_edge(newNodeID, nodeName)

                    # Reopen the node if it was closed
                    if existingNode.get("status") == "closed":
                        self.graph.nodes[newNodeID]["status"] = 'open'
                        self._insertNodeNameInOpenList(newNodeID)
                    # If it's already open, add it again to heap with better f-value
                    # (the old entry will be filtered out by _getBestNodeName)
                    elif existingNode.get("status") == "open":
                        self._insertNodeNameInOpenList(newNodeID)

        return []

    # wird aktuell nicht verwendet
    @IPPerfMonitor
    def _handleNode9(self, nodeName):
        """Generates possible successor positions also in diagonal direction with reopening support"""
        node = self.graph.nodes[nodeName]
        for i in range(len(node["pos"])):
            for j in range(len(node["pos"])):
                for u in [-1, 1]:
                    for v in [-1, 0, 1]:
                        newPos = copy.copy(node["pos"])
                        newPos[i] += u * self.step_size[i]
                        newPos[j] += v * self.step_size[j]

                        # Check if position is within limits
                        if not self._inLimits(newPos):
                            continue

                        # Edge collision check
                        if self.checkEdgeCollision:
                            if self._collisionChecker.lineInCollision(node["pos"], newPos):
                                continue

                        newNodeID = self._getNodeID(newPos)
                        newG = node["g"] + euclidean(node["pos"], newPos)

                        # Case 1: Node doesn't exist yet - create it
                        if newNodeID not in self.graph:
                            self._addGraphNode(newPos, nodeName)
                            continue

                        # Case 2: Node exists - check if we should update it (reopening)
                        existingNode = self.graph.nodes[newNodeID]

                        # If reopening is disabled and node exists, skip it (standard A* behavior)
                        if not self.allowReopening:
                            continue

                        # If reopening is enabled, check if we found a better path
                        if newG < existingNode["g"]:
                            # Update the node with better cost
                            self.graph.nodes[newNodeID]["g"] = newG

                            # Remove old parent edge(s) and add new one
                            # Graph structure: child -> parent (stored as successor)
                            oldParents = list(self.graph.successors(newNodeID))
                            if oldParents:
                                self.graph.remove_edges_from([(newNodeID, p) for p in oldParents])

                            # Add new parent edge: child -> parent
                            self.graph.add_edge(newNodeID, nodeName)

                            # Reopen the node if it was closed
                            if existingNode.get("status") == "closed":
                                self.graph.nodes[newNodeID]["status"] = 'open'
                                self._insertNodeNameInOpenList(newNodeID)
                            # If it's already open, add it again to heap with better f-value
                            # (the old entry will be filtered out by _getBestNodeName)
                            elif existingNode.get("status") == "open":
                                self._insertNodeNameInOpenList(newNodeID)

        return []
