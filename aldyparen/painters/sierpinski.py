from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from .base import Painter


# If point is inside, returns 0.
# If point is outside, returns at which iteration we exited it.
# i8(f8,f8,i8)
@numba.jit(nopython=True)
def is_point_outside_carpet(x: float, y: float, depth: int) -> int:
    if depth <= 0:
        return 0
    x *= 3
    y *= 3
    kx = int(np.floor(x))
    ky = int(np.floor(y))
    if kx == 1 and ky == 1:
        return 1
    return is_point_outside_carpet(x - kx, y - ky, depth - 1)


# u4(c16,i8)
@numba.vectorize(nopython=True)
def sierpinski_numba(p: Any, depth: Any) -> int:
    x = np.real(p)
    y = np.imag(p)
    if 0 <= x < 1 and 0 <= y < 1:
        return 1 - is_point_outside_carpet(x, y, depth)
    else:
        return 0


class SierpinskiCarpetPainter(Painter):
    def __init__(self, depth: int = 3):
        self.depth = depth

    def to_object(self) -> dict[str, Any]:
        return {"depth": self.depth}

    def paint(self, points: NDArray[np.complex128], ans: NDArray[np.uint32]) -> None:
        ans[:] = sierpinski_numba(points, self.depth)
