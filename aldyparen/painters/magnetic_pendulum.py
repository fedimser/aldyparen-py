import math
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from .base import Painter


DEFAULT_MAGNETS = [
    {"x": -1.0, "y": -0.58, "strength": 1.0},
    {"x": 1.0, "y": -0.58, "strength": 1.0},
    {"x": 0.0, "y": 1.15, "strength": 1.0},
]
SETTLING_TIME_BINS = 16


@numba.njit(parallel=True)
def paint_magnetic_pendulum_numba(
    points: NDArray[np.complex128],
    ans: NDArray[np.uint32],
    magnet_positions: NDArray[np.float64],
    magnet_strengths: NDArray[np.float64],
    height: float,
    damping: float,
    gravity: float,
    time_step: float,
    max_steps: int,
    settle_distance: float,
    settle_speed: float,
) -> int:
    numerical_failures = 0
    magnet_count = len(magnet_strengths)
    height_squared = height * height
    settle_distance_squared = settle_distance * settle_distance
    settle_speed_squared = settle_speed * settle_speed

    for point_index in numba.prange(len(points)):
        point = points[point_index]
        x = point.real
        y = point.imag
        velocity_x = 0.0
        velocity_y = 0.0
        ans[point_index] = 0
        valid = np.isfinite(x) and np.isfinite(y)

        for step in range(max_steps):
            nearest_magnet = -1
            nearest_distance_squared = np.inf
            for magnet_index in range(magnet_count):
                dx = magnet_positions[magnet_index, 0] - x
                dy = magnet_positions[magnet_index, 1] - y
                distance_squared = dx * dx + dy * dy
                if distance_squared < nearest_distance_squared:
                    nearest_distance_squared = distance_squared
                    nearest_magnet = magnet_index

            speed_squared = velocity_x * velocity_x + velocity_y * velocity_y
            if nearest_distance_squared <= settle_distance_squared and speed_squared <= settle_speed_squared:
                time_bin = min((step * SETTLING_TIME_BINS) // max_steps, SETTLING_TIME_BINS - 1)
                ans[point_index] = np.uint32(1 + nearest_magnet + magnet_count * time_bin)
                break

            acceleration_x = -gravity * x - damping * velocity_x
            acceleration_y = -gravity * y - damping * velocity_y
            for magnet_index in range(magnet_count):
                dx = magnet_positions[magnet_index, 0] - x
                dy = magnet_positions[magnet_index, 1] - y
                distance_squared = dx * dx + dy * dy + height_squared
                force_scale = magnet_strengths[magnet_index] / (distance_squared * math.sqrt(distance_squared))
                acceleration_x += force_scale * dx
                acceleration_y += force_scale * dy

            if not np.isfinite(acceleration_x) or not np.isfinite(acceleration_y):
                valid = False
                break

            # Semi-implicit Euler is stable for damped systems and avoids the
            # energy growth of ordinary forward Euler at the same step size.
            velocity_x += acceleration_x * time_step
            velocity_y += acceleration_y * time_step
            x += velocity_x * time_step
            y += velocity_y * time_step
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(velocity_x) and np.isfinite(velocity_y)):
                valid = False
                break

        if not valid:
            ans[point_index] = 0
            numerical_failures += 1

    return numerical_failures


class MagneticPendulumPainter(Painter):
    def __init__(
        self,
        magnets: list[dict[str, Any]] | None = None,
        height: float = 0.2,
        damping: float = 0.2,
        gravity: float = 0.5,
        time_step: float = 0.02,
        max_steps: int = 1200,
        settle_distance: float = 0.12,
        settle_speed: float = 0.02,
    ):
        magnets = DEFAULT_MAGNETS if magnets is None else magnets
        if not isinstance(magnets, list) or len(magnets) < 3:
            raise ValueError("magnets must be a list containing at least three magnets")

        normalized_magnets: list[dict[str, float]] = []
        for index, magnet in enumerate(magnets):
            if not isinstance(magnet, dict):
                raise ValueError(f"magnet {index} must be an object with x, y, and strength")
            if set(magnet) != {"x", "y", "strength"}:
                raise ValueError(f"magnet {index} must contain exactly x, y, and strength")
            normalized: dict[str, float] = {}
            for field in ("x", "y", "strength"):
                value = magnet[field]
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError(f"magnet {index} {field} must be a finite number")
                normalized[field] = float(value)
            if normalized["strength"] <= 0:
                raise ValueError(f"magnet {index} strength must be positive")
            normalized_magnets.append(normalized)

        self.height = self._positive_number("height", height)
        self.damping = self._non_negative_number("damping", damping)
        self.gravity = self._non_negative_number("gravity", gravity)
        self.time_step = self._positive_number("time_step", time_step)
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self.settle_distance = self._positive_number("settle_distance", settle_distance)
        self.settle_speed = self._non_negative_number("settle_speed", settle_speed)
        self.magnets = normalized_magnets
        self.warning = None
        self._magnet_positions = np.array([[magnet["x"], magnet["y"]] for magnet in self.magnets], dtype=np.float64)
        self._magnet_strengths = np.array([magnet["strength"] for magnet in self.magnets], dtype=np.float64)

    @staticmethod
    def _positive_number(name: str, value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    @staticmethod
    def _non_negative_number(name: str, value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number")
        return float(value)

    def to_object(self) -> dict[str, Any]:
        return {
            "magnets": [magnet.copy() for magnet in self.magnets],
            "height": self.height,
            "damping": self.damping,
            "gravity": self.gravity,
            "time_step": self.time_step,
            "max_steps": self.max_steps,
            "settle_distance": self.settle_distance,
            "settle_speed": self.settle_speed,
        }

    def paint(self, points: NDArray[np.complex128], ans: NDArray[np.uint32]) -> None:
        self.warning = None
        try:
            numerical_failures = paint_magnetic_pendulum_numba(
                points,
                ans,
                self._magnet_positions,
                self._magnet_strengths,
                self.height,
                self.damping,
                self.gravity,
                self.time_step,
                self.max_steps,
                self.settle_distance,
                self.settle_speed,
            )
        except Exception as error:
            ans.fill(0)
            self.warning = f"Error simulating magnetic pendulum: {error}"
            return

        if numerical_failures:
            self.warning = f"Could not simulate magnetic pendulum for {numerical_failures} point(s)."
