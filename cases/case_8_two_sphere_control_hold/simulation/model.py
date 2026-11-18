"""Certified path planning and current allocation for the Case 8 demo."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np

from ..model import B_HOLD, I_MAX, Z_PLANE
from ..model_aligned_bz import F_HOLD, F_MOVE, point_actuation_map


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE1_JSON = PROJECT_ROOT / "Results" / "results_case8_aligned_bz.json"
STAGE2_JSON = PROJECT_ROOT / "Results" / "results_case8_aligned_bz_stage2.json"

MAGNET1_HOME = np.array([0.02, 0.0], dtype=float)
MAGNET2_HOME = np.array([-0.02, 0.0], dtype=float)
MAGNET2_START = np.array([-0.04, -0.03], dtype=float)


def _string_pull(portals):
    """Taut path through an ordered list of ``(left, right)`` portals.

    This is the funnel algorithm.  "Left" is the counter-clockwise end of a
    portal as seen when travelling through it.  The first and last portals
    are the start and the goal, each given as a degenerate portal.
    """

    def area2(a, b, c):
        # Positive when c lies clockwise of the ray a -> b.
        return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])

    def same(a, b):
        return abs(a[0] - b[0]) < 1e-15 and abs(a[1] - b[1]) < 1e-15

    apex, left, right = portals[0][0], portals[0][0], portals[0][1]
    apex_i = left_i = right_i = 0
    points = [apex]
    i = 1
    budget = 4 * len(portals) ** 2 + 16
    while i < len(portals):
        budget -= 1
        if budget < 0:
            raise RuntimeError("funnel algorithm did not terminate")
        new_left, new_right = portals[i]
        if area2(apex, right, new_right) <= 0.0:
            if same(apex, right) or area2(apex, left, new_right) > 0.0:
                right, right_i = new_right, i
            else:
                points.append(left)
                apex, apex_i = left, left_i
                left, right, left_i, right_i = apex, apex, apex_i, apex_i
                i = apex_i + 1
                continue
        if area2(apex, left, new_left) >= 0.0:
            if same(apex, left) or area2(apex, right, new_left) < 0.0:
                left, left_i = new_left, i
            else:
                points.append(right)
                apex, apex_i = right, right_i
                left, right, left_i, right_i = apex, apex, apex_i, apex_i
                i = apex_i + 1
                continue
        i += 1
    points.append(portals[-1][0])

    compact = [np.asarray(points[0], dtype=float)]
    for point in points[1:]:
        point = np.asarray(point, dtype=float)
        if np.linalg.norm(point - compact[-1]) > 1e-12:
            compact.append(point)
    return np.asarray(compact)


class CertifiedSlice:
    """A certified 2D TFW represented by rectangles and their Bz witnesses.

    The adjacency graph connects rectangles only across a shared edge of
    positive length. :meth:`plan` passes through the midpoint of each shared
    edge, so consecutive route segments lie in a certified rectangle.
    :meth:`plan_taut` pulls that route tight within the same rectangles.
    Neither planner cuts across an undetermined or outside gap.
    """

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not data.get("complete", False):
            raise ValueError(f"TFW result is incomplete: {self.path}")
        boxes = data.get("in_workspace", [])
        witnesses = data.get("inside_witnesses", [])
        if not boxes or len(boxes) != len(witnesses):
            raise ValueError(f"Missing or inconsistent certified boxes: {self.path}")
        self.boxes = np.asarray(boxes, dtype=float)
        self.witnesses = np.asarray([item["Bz_T"] for item in witnesses], dtype=float)
        witness_boxes = np.asarray([item["box"] for item in witnesses], dtype=float)
        if not np.allclose(self.boxes, witness_boxes, rtol=0.0, atol=1e-13):
            raise ValueError(f"Witness order does not match boxes: {self.path}")
        self.undetermined = np.asarray(data.get("undetermined", []), dtype=float).reshape(-1, 4)
        self.outside = np.asarray(data.get("out_of_workspace", []), dtype=float).reshape(-1, 4)
        self.held = np.asarray(data["protocol"]["held_position_m"][:2], dtype=float)
        self.moments = tuple(data["protocol"]["moments_A_m2"])
        if not math.isclose(self.moments[0], self.moments[1], rel_tol=0.0, abs_tol=1e-14):
            raise ValueError("Sequential role exchange requires identical magnets")
        self._graph = None

    @property
    def centers(self):
        b = self.boxes
        return np.column_stack(((b[:, 0] + b[:, 1]) / 2, (b[:, 2] + b[:, 3]) / 2))

    def containing_indices(self, point, tol=2e-12):
        x, y = np.asarray(point, dtype=float)
        b = self.boxes
        mask = (
            (b[:, 0] - tol <= x) & (x <= b[:, 1] + tol)
            & (b[:, 2] - tol <= y) & (y <= b[:, 3] + tol)
        )
        return np.flatnonzero(mask)

    def contains(self, point):
        return self.containing_indices(point).size > 0

    def reach(self, start, end):
        """Largest ``t`` in ``[0, 1]`` such that the segment from ``start`` to
        ``start + t (end - start)`` lies in the certified union.

        The segment is followed one rectangle at a time: from the current
        point it advances to the furthest exit among the rectangles that
        contain the point, so every piece lies in one convex rectangle.
        Returns 0 when ``start`` itself is not certified.
        """

        start = np.asarray(start, dtype=float)
        delta = np.asarray(end, dtype=float) - start
        t = 0.0
        for _ in range(len(self.boxes) + 1):
            indices = self.containing_indices(start + t * delta)
            if indices.size == 0:
                return t
            b = self.boxes[indices]
            with np.errstate(divide="ignore", invalid="ignore"):
                exit_x = np.where(delta[0] > 0, (b[:, 1] - start[0]) / delta[0],
                                  np.where(delta[0] < 0, (b[:, 0] - start[0]) / delta[0],
                                           np.inf))
                exit_y = np.where(delta[1] > 0, (b[:, 3] - start[1]) / delta[1],
                                  np.where(delta[1] < 0, (b[:, 2] - start[1]) / delta[1],
                                           np.inf))
            exit_t = float(np.max(np.minimum(exit_x, exit_y)))
            if exit_t >= 1.0:
                return 1.0
            if exit_t <= t:
                return t
            t = exit_t
        return t

    def box_and_witness(self, point):
        indices = self.containing_indices(point)
        if indices.size == 0:
            raise ValueError("position is outside the certified TFW")
        # Prefer the smallest containing box at shared boundaries.
        areas = ((self.boxes[indices, 1] - self.boxes[indices, 0])
                 * (self.boxes[indices, 3] - self.boxes[indices, 2]))
        index = int(indices[np.argmin(areas)])
        return index, self.witnesses[index]

    @staticmethod
    def _key(value):
        return round(float(value), 12)

    def _build_graph(self):
        n = len(self.boxes)
        graph = [[] for _ in range(n)]
        left, right, bottom, top = {}, {}, {}, {}
        for index, (x0, x1, y0, y1) in enumerate(self.boxes):
            left.setdefault(self._key(x0), []).append((index, y0, y1))
            right.setdefault(self._key(x1), []).append((index, y0, y1))
            bottom.setdefault(self._key(y0), []).append((index, x0, x1))
            top.setdefault(self._key(y1), []).append((index, x0, x1))

        def connect(group_a, group_b, vertical):
            for coordinate in group_a.keys() & group_b.keys():
                for ia, a0, a1 in group_a[coordinate]:
                    for ib, b0, b1 in group_b[coordinate]:
                        lo, hi = max(a0, b0), min(a1, b1)
                        if hi - lo <= 1e-13:
                            continue
                        middle = (lo + hi) / 2
                        if vertical:
                            portal = (coordinate, middle)
                            ends = ((coordinate, lo), (coordinate, hi))
                        else:
                            portal = (middle, coordinate)
                            ends = ((lo, coordinate), (hi, coordinate))
                        graph[ia].append((ib, portal, ends))
                        graph[ib].append((ia, portal, ends))

        connect(right, left, True)
        connect(top, bottom, False)
        self._graph = graph

    def _search(self, start, goal):
        """A* over the rectangle adjacency graph.

        Returns ``None`` when one rectangle holds both points.  Otherwise
        returns the rectangle sequence and, for each step along it, the shared
        edge crossed, as ``(midpoint, (low_end, high_end))``.
        """

        starts = self.containing_indices(start)
        goals = set(map(int, self.containing_indices(goal)))
        if starts.size == 0:
            raise ValueError("current position is outside the certified TFW")
        if not goals:
            raise ValueError("requested destination is outside the certified TFW")
        if goals.intersection(map(int, starts)):
            return None
        if self._graph is None:
            self._build_graph()

        centers = self.centers
        queue = []
        distance = {}
        parent = {}
        parent_edge = {}
        for index in map(int, starts):
            distance[index] = 0.0
            parent[index] = None
            heuristic = float(np.linalg.norm(centers[index] - goal))
            heapq.heappush(queue, (heuristic, 0.0, index))

        reached = None
        while queue:
            _, cost, index = heapq.heappop(queue)
            if cost != distance.get(index):
                continue
            if index in goals:
                reached = index
                break
            for neighbor, portal, ends in self._graph[index]:
                step = float(np.linalg.norm(centers[index] - np.asarray(portal)))
                step += float(np.linalg.norm(centers[neighbor] - np.asarray(portal)))
                new_cost = cost + step
                if new_cost + 1e-15 >= distance.get(neighbor, math.inf):
                    continue
                distance[neighbor] = new_cost
                parent[neighbor] = index
                parent_edge[neighbor] = (portal, ends)
                heuristic = float(np.linalg.norm(centers[neighbor] - goal))
                heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
        if reached is None:
            raise ValueError("destination is in a disconnected certified component")

        nodes = []
        cursor = reached
        while cursor is not None:
            nodes.append(cursor)
            cursor = parent[cursor]
        nodes.reverse()
        return nodes, [parent_edge[node] for node in nodes[1:]]

    def plan(self, start, goal):
        """Return a certified portal path from ``start`` to ``goal``."""

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        route = self._search(start, goal)
        if route is None:
            return np.vstack((start, goal))
        _, edges = route
        points = [start]
        for portal, _ in edges:
            points.append(np.asarray(portal, dtype=float))
        points.append(goal)

        compact = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - compact[-1]) > 1e-12:
                compact.append(point)
        path = np.asarray(compact)
        if not self.path_is_certified(path):
            raise RuntimeError("internal error: planned segment left certified boxes")
        return path

    def plan_taut(self, start, goal):
        """Shortest path from ``start`` to ``goal`` through the rectangles of
        :meth:`plan`.

        The midpoint route zig-zags across the small rectangles.  Pulling it
        taut keeps each crossing of a shared edge inside that edge, so the
        piece between consecutive crossings lies in one convex rectangle.  The
        result is re-checked segment by segment with :meth:`reach`; should the
        check fail, the midpoint route is returned instead.
        """

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        route = self._search(start, goal)
        if route is None:
            return np.vstack((start, goal))
        nodes, edges = route
        portals = [(start, start)]
        for box, (_, (low, high)) in zip(nodes[:-1], edges):
            low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
            x0, x1, y0, y1 = self.boxes[box]
            if low[0] == high[0]:
                # Vertical edge: travelling +x, the high end is on the left.
                forward = abs(x1 - low[0]) < 1e-9
                portals.append((high, low) if forward else (low, high))
            else:
                # Horizontal edge: travelling +y, the low end is on the left.
                forward = abs(y1 - low[1]) < 1e-9
                portals.append((low, high) if forward else (high, low))
        portals.append((goal, goal))
        try:
            path = _string_pull(portals)
        except RuntimeError:
            return self.plan(start, goal)
        if len(path) >= 2 and all(
            self.reach(a, b) >= 1.0 for a, b in zip(path[:-1], path[1:])
        ):
            return path
        return self.plan(start, goal)

    def path_is_certified(self, path):
        """Check that every segment lies in one certified convex rectangle."""

        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return False
        for start, end in zip(path[:-1], path[1:]):
            a = set(map(int, self.containing_indices(start)))
            b = set(map(int, self.containing_indices(end)))
            if not a.intersection(b):
                return False
        return True


def solve_control(slice_data, moving_position, moving_force, held_force=(0.0, 0.0)):
    """Allocate currents using the certified Bz witness at the current point."""

    moving_position = np.asarray(moving_position, dtype=float)
    moving_force = np.asarray(moving_force, dtype=float)
    held_force = np.asarray(held_force, dtype=float)
    if np.max(np.abs(moving_force)) > F_MOVE + 1e-12:
        raise ValueError("moving force exceeds the certified component bound")
    if np.max(np.abs(held_force)) > F_HOLD + 1e-12:
        raise ValueError("held force exceeds the certified component bound")
    _, (bz_held, bz_moving) = slice_data.box_and_witness(moving_position)
    G, offset = point_actuation_map(
        moving_position[0], moving_position[1], bz_held, bz_moving,
        held_x=slice_data.held[0], held_y=slice_data.held[1],
    )
    task = np.array([
        0.0, B_HOLD, bz_held, held_force[0], held_force[1],
        0.0, B_HOLD, bz_moving, moving_force[0], moving_force[1],
    ])
    currents = np.linalg.solve(G, task - offset)
    residual = G @ currents + offset - task
    return {
        "currents": currents,
        "peak_current": float(np.max(np.abs(currents))),
        "residual": residual,
        "bz": (float(bz_held), float(bz_moving)),
        "within_limits": bool(np.max(np.abs(currents)) <= I_MAX + 1e-7),
    }


def self_check():
    stage1 = CertifiedSlice(STAGE1_JSON)
    stage2 = CertifiedSlice(STAGE2_JSON)
    path1 = stage1.plan(MAGNET2_START, MAGNET2_HOME)
    path2 = stage2.plan(MAGNET1_HOME, MAGNET1_HOME)
    assert stage1.contains(MAGNET2_HOME)
    assert stage2.contains(MAGNET1_HOME)
    assert len(path1) >= 2 and len(path2) == 2
    result = solve_control(stage1, MAGNET2_START, (0.0, 0.0))
    assert result["within_limits"]
    print(f"stage-1 return waypoints: {len(path1)}")
    print(f"zero-force peak current: {result['peak_current']:.6f} A")
    print("case 8 sequential simulation model: OK")


if __name__ == "__main__":
    self_check()
