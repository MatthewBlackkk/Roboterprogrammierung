# python
import copy
import heapq
import networkx as nx
import math
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

        self.goal = []
        self.goalFound = False

        self.limits = self._collisionChecker.getEnvironmentLimits()

        # Bei hochsetzen der stepsize muss entsprechend die break number angepasst werden
        #self.num_steps = [44, 75]  # Unterschiedliche Diskretisierung für x und y
        #self.step_size = []
        #for i, limit in enumerate(self.limits):
        #    self.step_size.append((limit[1] - limit[0]) / self.num_steps[i])
        
        self.num_steps = None
        self.step_size = None
        self.allowReopening = False

        self.w = 0.5
        return
    
    def _getNodeID(self, pos):
        nodeId = "-"
        for i in pos:
            # Round to avoid float precision problems
            nodeId += str(round(i, 4)) + "-"
        return nodeId
    
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
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options

        Example:

            config["w"] = 0.5
            config["heuristic"] = "euclid"

        """
        # 0. reset
        self.graph.clear()

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
                self.step_size.append((limit[1] - limit[0]) / self.num_steps[i])
            

            # Erweiterung für Kantenkollision -- Ludwig
            # Erklärung: config.get(checkEdgeCollision,False) liest den Wert aus dem config-Dictionary aus.
            # Falls er nicht vorhanden ist, wird standardmäßig False verwendet.
            self.checkEdgeCollision = config.get("checkEdgeCollision", False)
            # Ende Erweiterung für Kantenkollision -- Ludwig

            self.goal = checkedGoalList[0]
            self._addGraphNode(checkedStartList[0])

            # acceptance_radius = min(self.step_size) * 0.9
            acceptance_radius = math.sqrt(sum([(s / 2.0) ** 2 for s in self.step_size])) * 1.1

            currentBestName = self._getBestNodeName()
            breakNumber = 0

            max_iterations = 100000
            while currentBestName:
                if breakNumber > max_iterations:
                    print(f"A* Warnung: Max. Iterationen ({max_iterations}) erreicht. Graph hat {self.graph.number_of_nodes()} Knoten.")
                    break

                breakNumber += 1

                currentBest = self.graph.nodes[currentBestName]

                dist_to_goal = euclidean(currentBest["pos"], self.goal)

                # check whether goal reached but not with == because of float precision
                if dist_to_goal < acceptance_radius:
                    self.solutionPath = []
                    self._collectPath(currentBestName, self.solutionPath)
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
        except Exception:
            print("Planning failed")
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

