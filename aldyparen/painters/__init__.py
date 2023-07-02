import warnings

import numba
import numpy as np

from .mandelbroid import MandelbroidPainter
from .mandelbrot_hp import MadnelbrotHighPrecisionPainter


# Contract:
# __init__ takes all args as kwargs. Should create default instanjce with no args.
# to_object produces the same kwargs as taken by __init__
# paint takes 2d array of complex points and paints them (think about how this works for arbitrary precision).

@numba.vectorize("u4(c16,f8)", target="parallel")
def fff(p, width):
    if abs(p.real) < width or abs(p.imag) < width or abs(p - (1 + 1j)) < width:
        return 1
    else:
        return 0


class Painter:
    pass


class AxisPainter(Painter):
    def __init__(self, width=0.01):
        self.width = width

    def to_object(self):
        return {"width": self.width}

    def paint(self, points):
        return fff(points, self.width)

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


class SierpinskiCarpetPainter(Painter):
    def __init__(self, depth=3):
        self.depth = depth

    def to_object(self):
        return {"depth": self.depth}

    def paint(self, points):
        ans = sierpinski_numba(points, self.depth)
        return ans


# All supported painters.
ALL_PAINTERS = [MandelbroidPainter, MadnelbrotHighPrecisionPainter, AxisPainter, SierpinskiCarpetPainter]
