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
        # Guard: return None if heap is empty
        if not self.openList:
            return None

        # Pop best; skip stale entries
        while self.openList:
            _, nodeName = heapq.heappop(self.openList)
            node = self.graph.nodes.get(nodeName)
            if node is None:
                continue
            if node.get("status") == "open":
                return nodeName
            # else: closed/outdated -> skip
        return None

    def _add_or_reopen(self, newPos, fatherName, move_cost=1.0):
        succ_id = self._getNodeID(newPos)

        # New node: add normally
        if succ_id not in self.graph:
            self.graph.add_node(succ_id, pos=newPos, status='open', g=0)
            if fatherName is not None:
                self.graph.add_edge(succ_id, fatherName)
                self.graph.nodes[succ_id]["g"] = self.graph.nodes[fatherName]["g"] + move_cost
            self._insertNodeNameInOpenList(succ_id)
            return

        # Existing node: reopen only if enabled and better g found
        if not self.reopen:
            return

        succ = self.graph.nodes[succ_id]
        tentative_g = self.graph.nodes[fatherName]["g"] + move_cost

        if tentative_g < succ["g"]:
            succ["g"] = tentative_g
            succ["status"] = "open"
            # Rewire parent edge: ensure single parent
            old_parents = list(self.graph.successors(succ_id))
            if old_parents:
                self.graph.remove_edges_from([(succ_id, p) for p in old_parents])
            self.graph.add_edge(succ_id, fatherName)
            self._insertNodeNameInOpenList(succ_id)

    @IPPerfMonitor
    def _handleNode(self, nodeName):
        """4-neighborhood with optional reopening."""
        node = self.graph.nodes[nodeName]
        pos = node["pos"]

        for i in range(len(pos)):
            for u in [-1, 1]:
                newPos = copy.copy(pos)
                newPos[i] += u
                if not self._inLimits(newPos):
                    continue

                # Axis-aligned move cost = 1
                self._add_or_reopen(newPos, nodeName, move_cost=1.0)

        return []

    @IPPerfMonitor
    def _handleNode9(self, nodeName):
        """8/9-neighborhood with optional reopening (diagonals allowed)."""
        node = self.graph.nodes[nodeName]
        pos = node["pos"]
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
                        # Skip zero move
                        if u == 0 and v == 0:
                            continue
                        # Cost: diagonal sqrt(2) when both offsets non-zero on different axes
                        is_diagonal = (i != j) and (u != 0) and (v != 0)
                        move_cost = math.sqrt(2) if is_diagonal else 1.0

                        self._add_or_reopen(newPos, nodeName, move_cost=move_cost)

        return []