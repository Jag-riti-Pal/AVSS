"""
Module: tripwire.py
Purpose: Planar vector geometry engine for boundary line-crossing detection.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


class VectorTripwire:
    def __init__(self, start_point: Tuple[int, int], end_point: Tuple[int, int]):
        self.p1 = Point2D(*start_point)
        self.p2 = Point2D(*end_point)
        # Persistent target trajectory history: {track_id: Point2D}
        self._trajectory_cache: Dict[int, Point2D] = {}
        
        # State counters
        self.ingress_count: int = 0
        self.egress_count: int = 0

    @staticmethod
    def _ccw(a: Point2D, b: Point2D, c: Point2D) -> float:
        """Evaluates determinant to determine orientation of (A, B, C)."""
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    def _is_intersecting(self, a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
        """Determines if line segment AB intersects segment CD."""
        d1 = self._ccw(c, d, a)
        d2 = self._ccw(c, d, b)
        d3 = self._ccw(a, b, c)
        d4 = self._ccw(a, b, d)

        return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
                ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))

    def update(self, track_id: int, bottom_center_anchor: Tuple[int, int]) -> Optional[str]:
        """
        Ingests the current centroid/anchor of an entity, evaluates trajectory intersection,
        and increments directional counters accordingly.
        """
        curr_pos = Point2D(*bottom_center_anchor)

        if track_id not in self._trajectory_cache:
            self._trajectory_cache[track_id] = curr_pos
            return None

        prev_pos = self._trajectory_cache[track_id]
        self._trajectory_cache[track_id] = curr_pos

        # Check for intersection between previous-current step and tripwire
        if self._is_intersecting(prev_pos, curr_pos, self.p1, self.p2):
            # Compute cross-product of tripwire vector x trajectory vector
            cross_product = (
                (self.p2.x - self.p1.x) * (curr_pos.y - prev_pos.y) -
                (self.p2.y - self.p1.y) * (curr_pos.x - prev_pos.x)
            )

            if cross_product > 0:
                self.ingress_count += 1
                return "INGRESS"
            elif cross_product < 0:
                self.egress_count += 1
                return "EGRESS"

        return None

    def purge_stale_tracks(self, active_track_ids: set) -> None:
        """Removes expired tracking entries to bound memory consumption."""
        stale_ids = [tid for tid in self._trajectory_cache if tid not in active_track_ids]
        for tid in stale_ids:
            del self._trajectory_cache[tid]

    def reset_counts(self) -> None:
        self.ingress_count = 0
        self.egress_count = 0
        self._trajectory_cache.clear()