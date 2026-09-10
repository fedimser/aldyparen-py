import math
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from .base import Painter


@numba.njit(parallel=True)
def paint_lyapunov_numba(
    points: NDArray[np.complex128],
    ans: NDArray[np.uint32],
    sequence: NDArray[np.uint8],
    warmup: int,
    iterations: int,
    color_scale: float,
) -> int:
    numerical_failures = 0
    sequence_length = len(sequence)
    for point_index in numba.prange(len(points)):
        point = points[point_index]
        parameter_a = point.real
        parameter_b = point.imag
        x = 0.5
        valid = np.isfinite(parameter_a) and np.isfinite(parameter_b)

        for iteration in range(warmup):
            parameter = parameter_a if sequence[iteration % sequence_length] == 0 else parameter_b
            x = parameter * x * (1.0 - x)
            if not np.isfinite(x):
                valid = False
                break

        exponent_sum = 0.0
        is_superstable = False
        if valid:
            for iteration in range(iterations):
                sequence_index = (warmup + iteration) % sequence_length
                parameter = parameter_a if sequence[sequence_index] == 0 else parameter_b
                derivative = abs(parameter * (1.0 - 2.0 * x))
                if derivative == 0.0:
                    is_superstable = True
                elif not np.isfinite(derivative):
                    valid = False
                    break
                elif not is_superstable:
                    exponent_sum += math.log(derivative)
                x = parameter * x * (1.0 - x)
                if not np.isfinite(x):
                    valid = False
                    break

        if not valid:
            ans[point_index] = 0
            numerical_failures += 1
            continue

        if is_superstable:
            ans[point_index] = np.uint32(0xFFFFFFFF)
            continue

        exponent = exponent_sum / iterations
        magnitude_bin = math.floor(abs(exponent) * color_scale)
        magnitude_bin = min(magnitude_bin, 0x7FFFFFFE)
        ans[point_index] = np.uint32(2 * magnitude_bin + (1 if exponent < 0.0 else 2))

    return numerical_failures


class LyapunovFractalPainter(Painter):
    def __init__(
        self,
        sequence: str = "AABAB",
        warmup: int = 100,
        iterations: int = 100,
        color_scale: float = 10.0,
    ):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("sequence must be a non-empty string containing only A and B")
        if any(symbol not in "AB" for symbol in sequence):
            raise ValueError("sequence must contain only A and B")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if (
            not isinstance(color_scale, (int, float))
            or isinstance(color_scale, bool)
            or not math.isfinite(color_scale)
            or color_scale <= 0
        ):
            raise ValueError("color_scale must be a positive finite number")

        self.sequence = sequence
        self.warmup = warmup
        self.iterations = iterations
        self.color_scale = float(color_scale)
        self.warning = None
        self._encoded_sequence = np.array([symbol == "B" for symbol in sequence], dtype=np.uint8)

    def to_object(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "color_scale": self.color_scale,
        }

    def paint(self, points: NDArray[np.complex128], ans: NDArray[np.uint32]) -> None:
        self.warning = None
        try:
            numerical_failures = paint_lyapunov_numba(
                points,
                ans,
                self._encoded_sequence,
                self.warmup,
                self.iterations,
                self.color_scale,
            )
        except Exception as error:
            ans.fill(0)
            self.warning = f"Error computing Lyapunov exponents: {error}"
            return

        if numerical_failures:
            self.warning = f"Could not compute Lyapunov exponents for {numerical_failures} point(s)."
