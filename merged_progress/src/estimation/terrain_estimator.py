"""Contact terrain estimator: builds a height map from foot touchdowns.

On each foot landing we record the foot-site z in world frame. Future CoM-height
references use the locally measured surface instead of a known analytic model.
This lets the controller adapt to unknown bumps, stairs, and slopes.
"""
import numpy as np


class ContactTerrainEstimator:
    """Incrementally builds a sparse height map from foot-contact events.

    The map is keyed by discretised (ix, iy) bucket indices.  When a height
    query falls outside any recorded bucket the call is forwarded to a fallback
    terrain object (if one has been set) or returns 0.0.
    """

    def __init__(self, bucket_xy: float = 0.08):
        self._map: dict = {}          # (ix, iy) -> float z
        self._bucket = bucket_xy      # spatial resolution [m]
        self._fallback = None         # terrain object with .height(x,y) / .surface_R(x,y)

    # ------------------------------------------------------------------
    def set_fallback(self, terrain) -> None:
        """Store a fallback terrain used when no contact data is available."""
        self._fallback = terrain

    # ------------------------------------------------------------------
    def record(self, x: float, y: float, z: float) -> None:
        """Record a foot touchdown at world position (x, y, z).

        Stores in the nearest bucket; also fills neighbouring buckets with
        the same z value (simple piecewise-constant interpolation that works
        well for flat, ramped, and staired terrain).
        """
        ix = round(x / self._bucket)
        iy = round(y / self._bucket)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                key = (ix + dx, iy + dy)
                if key not in self._map:          # don't overwrite closer measurements
                    self._map[key] = z
        self._map[(ix, iy)] = z                   # own bucket always updated

    # ------------------------------------------------------------------
    def height(self, x: float, y: float) -> float:
        """Return estimated terrain height at (x, y)."""
        ix = round(x / self._bucket)
        iy = round(y / self._bucket)
        if (ix, iy) in self._map:
            return self._map[(ix, iy)]
        if self._fallback is not None:
            return self._fallback.height(x, y)
        return 0.0

    # ------------------------------------------------------------------
    def surface_R(self, x: float, y: float) -> np.ndarray:
        """Estimate surface orientation.

        Currently returns identity (flat assumption per bucket).
        Could be extended to compute a local gradient from neighbouring
        bucket heights.
        """
        if self._fallback is not None:
            return self._fallback.surface_R(x, y)
        return np.eye(3)

    # ------------------------------------------------------------------
    @property
    def n_contacts(self) -> int:
        """Number of unique buckets that have been populated by touchdowns."""
        return len(self._map)
