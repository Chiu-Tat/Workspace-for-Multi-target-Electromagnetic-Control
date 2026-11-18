"""A* on a boolean raster of certified TFW cells, plus shortcut smoothing.

Used on precomputed TFW rasters only; nothing here certifies feasibility.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def astar(free, start, goal):
    """8-connected A* on ``free[row, col]``; start/goal are (row, col)."""
    rows, cols = free.shape
    if not (free[start] and free[goal]):
        return None
    g = {start: 0.0}
    parent = {start: None}
    heap = [(0.0, start)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dr, dc in _MOVES:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or not free[nxt]:
                continue
            cost = g[cur] + math.hypot(dr, dc)
            if cost < g.get(nxt, math.inf):
                g[nxt] = cost
                parent[nxt] = cur
                h = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(heap, (cost + h, nxt))
    return None


def segment_free(free, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for t in np.linspace(0.0, 1.0, n):
        r = int(round(a[0] + t * (b[0] - a[0])))
        c = int(round(a[1] + t * (b[1] - a[1])))
        if not free[r, c]:
            return False
    return True


def shortcut(free, path):
    """Greedy line-of-sight shortcut; the result stays in ``free``."""
    if not path:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_free(free, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def resample(points, speed, dt):
    """Constant-speed resampling of a polyline; returns (t, points)."""
    p = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(2, int(math.ceil(total / (speed * dt))) + 1)
    si = np.linspace(0.0, total, n)
    out = np.column_stack([np.interp(si, s, p[:, k]) for k in range(p.shape[1])])
    return si / speed, out
