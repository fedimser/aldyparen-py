import numba
import numpy as np


# If point is inside, returns 0.
# If point is outside, returns at which iteration we exited it.
@numba.jit("u4(f8,f8,i8)")
def is_point_outside_carpet(x, y, depth):
    if depth <= 0:
        return 0
    x *= 3
    y *= 3
    kx = int(np.floor(x))
    ky = int(np.floor(y))
    if kx == 1 and ky == 1:
        return 1
    return is_point_outside_carpet(x - kx, y - ky, depth - 1)


@numba.vectorize("u4(c16,i8)", target="parallel")
def sierpinski_numba(p, depth):
    x = np.real(p)
    y = np.imag(p)
    if 0 <= x < 1 and 0 <= y < 1:
        return 1 - is_point_outside_carpet(x, y, depth)
    else:
        return 0


class SierpinskiCarpetPainter:
    def __init__(self, depth=3):
        self.depth = depth

    def to_object(self):
        return {"depth": self.depth}

    def paint(self, points, ans):
        ans[:] = sierpinski_numba(points, self.depth)
