# python

# python
import copy
import heapq
import math
from IPAStar import AStar

from IPPerfMonitor import IPPerfMonitor


class ReopenAStar(AStar):
    def __init__(self, collChecker=0, reopen=True):
        super(ReopenAStar, self).__init__(collChecker)
        self.reopen = reopen

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
        """4-neighborhood with optional reopening."""
        node = self.graph.nodes[nodeName]
        pos = node["pos"]
        current_g = node["g"]

        for i in range(len(pos)):
            for u in [-1, 1]:
                newPos = copy.copy(pos)
                newPos[i] += u
                if not self._inLimits(newPos):
                    continue

                succ_id = self._getNodeID(newPos)
                tentative_g = current_g + 1.0

                if succ_id not in self.graph:
                    self._addGraphNode(newPos, nodeName)
                    continue

                if not self.reopen:
                    continue

                succ = self.graph.nodes[succ_id]
                if tentative_g < succ["g"]:
                    succ["g"] = tentative_g
                    succ["status"] = "open"
                    # Rewire parent to current node
                    old_parents = list(self.graph.successors(succ_id))
                    if old_parents:
                        self.graph.remove_edges_from([(succ_id, p) for p in old_parents])
                    self.graph.add_edge(succ_id, nodeName)
                    self._insertNodeNameInOpenList(succ_id)
        return []

    @IPPerfMonitor
    def _handleNode9(self, nodeName):
        """8/9-neighborhood with optional reopening (diagonals allowed)."""
        node = self.graph.nodes[nodeName]
        pos = node["pos"]
        current_g = node["g"]
        dim = len(pos)

        for i in range(dim):
            for j in range(dim):
                for u in [-1, 1]:
                    for v in [-1, 0, 1]:
                        newPos = copy.copy(pos)
                        newPos[i] += u
                        newPos[j] += v
                        if not self._inLimits(newPos):
                            continue

                        succ_id = self._getNodeID(newPos)
                        is_diagonal = (i != j) and (u != 0) and (v != 0)
                        move_cost = math.sqrt(2) if is_diagonal else 1.0
                        tentative_g = current_g + move_cost

                        if succ_id not in self.graph:
                            self._addGraphNode(newPos, nodeName)
                            continue

                        if not self.reopen:
                            continue

                        succ = self.graph.nodes[succ_id]
                        if tentative_g < succ["g"]:
                            succ["g"] = tentative_g
                            succ["status"] = "open"
                            old_parents = list(self.graph.successors(succ_id))
                            if old_parents:
                                self.graph.remove_edges_from([(succ_id, p) for p in old_parents])
                            self.graph.add_edge(succ_id, nodeName)
                            self._insertNodeNameInOpenList(succ_id)
        return []