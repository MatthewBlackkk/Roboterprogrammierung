# python
import copy
import heapq
import networkx as nx
import math
from IPAStar import AStar
from scipy.spatial.distance import euclidean
from IPPerfMonitor import IPPerfMonitor

class ReopenAStar(AStar):
    def __init__(self, collChecker=0, reopen=True):
        super(ReopenAStar, self).__init__(collChecker)
        self.reopen = reopen

        self.graph = nx.DiGraph()  # = CloseList
        self.openList = []  # (<value>, <node>)

        self.goal = []
        self.goalFound = False

        self.limits = self._collisionChecker.getEnvironmentLimits()

        # Bei hochsetzen der stepsize muss entsprechend die break number angepasst werden
        self.num_steps = [44, 75]  # Unterschiedliche Diskretisierung für x und y
        self.step_size = []
        for i, limit in enumerate(self.limits):
            self.step_size.append((limit[1] - limit[0]) / self.num_steps[i])

        self.w = 0.5
        return

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
            while currentBestName:
                if breakNumber > 10000:
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
        except:
            print("Planning failed")
            return None

    def _getBestNodeName(self):
        # Guard against empty heap; skip stale entries
        if not self.openList:
            return None
        while self.openList:
            _, nodeName = heapq.heappop(self.openList)
            node = self.graph.nodes.get(nodeName)
            if node is None:
                continue
            if node.get("status") == "open":
                return nodeName
        return None

    @IPPerfMonitor
    def _handleNode(self, nodeName):
        """Generates successors in axis directions with optional reopening."""
        node = self.graph.nodes[nodeName]
        pos = node["pos"]
        current_g = node["g"]

        for i in range(len(pos)):
            for u in [-1, 1]:
                newPos = copy.copy(pos)
                newPos[i] += u * self.step_size[i]
                if not self._inLimits(newPos):
                    continue

                # Edge collision check like base class
                if self.checkEdgeCollision:
                    if self._collisionChecker.lineInCollision(pos, newPos):
                        continue

                succ_id = self._getNodeID(newPos)
                # If node is new: add normally
                if succ_id not in self.graph:
                    self._addGraphNode(newPos, nodeName)
                    continue

                # Base behavior: do not reopen if disabled
                if not self.reopen:
                    continue

                # Optional reopening with Euclidean cost
                succ = self.graph.nodes[succ_id]
                move_cost = euclidean(pos, newPos)
                tentative_g = current_g + move_cost
                if tentative_g < succ["g"]:
                    succ["g"] = tentative_g
                    succ["status"] = "open"
                    # Rewire single parent to current
                    old_parents = list(self.graph.successors(succ_id))
                    if old_parents:
                        self.graph.remove_edges_from([(succ_id, p) for p in old_parents])
                    self.graph.add_edge(succ_id, nodeName)
                    # Push; duplicates are fine
                    self._insertNodeNameInOpenList(succ_id)
        return []

    @IPPerfMonitor
    def _handleNode9(self, nodeName):
        """Generates successors incl. diagonals with optional reopening."""
        node = self.graph.nodes[nodeName]
        pos = node["pos"]
        current_g = node["g"]

        for i in range(len(pos)):
            for j in range(len(pos)):
                for u in [-1, 1]:
                    for v in [-1, 0, 1]:
                        newPos = copy.copy(pos)
                        newPos[i] += u * self.step_size[i]
                        newPos[j] += v * self.step_size[j]
                        if not self._inLimits(newPos):
                            continue

                        # Edge collision check like base class
                        if self.checkEdgeCollision:
                            if self._collisionChecker.lineInCollision(pos, newPos):
                                continue

                        succ_id = self._getNodeID(newPos)
                        if succ_id not in self.graph:
                            self._addGraphNode(newPos, nodeName)
                            continue

                        if not self.reopen:
                            continue

                        succ = self.graph.nodes[succ_id]
                        move_cost = euclidean(pos, newPos)
                        tentative_g = current_g + move_cost
                        if tentative_g < succ["g"]:
                            succ["g"] = tentative_g
                            succ["status"] = "open"
                            old_parents = list(self.graph.successors(succ_id))
                            if old_parents:
                                self.graph.remove_edges_from([(succ_id, p) for p in old_parents])
                            self.graph.add_edge(succ_id, nodeName)
                            self._insertNodeNameInOpenList(succ_id)
        return []
