import numba
import numpy as np

from .julia import JuliaPainter
from .mandelbroid import MandelbroidPainter
from .mandelbrot_hp import MadnelbrotHighPrecisionPainter


class Painter:
    """Abstract class for all painters.

    __init__ must take all args as kwargs. It must create default instance with no args.
    """

    @staticmethod
    def deserialize(class_name: str, data: dict) -> 'Painter':
        assert class_name in PAINTERS_INDEX
        painter_class = ALL_PAINTERS[PAINTERS_INDEX[class_name]]
        return painter_class(**data)

    def paint(self, points: np.ndarray, ans: np.ndarray) -> np.ndarray:
        """Paints given points. Must be implemented (unless this is a high-precision painter).

        :param points: Points to be painted (1D np.array of np.complex128).
        :param ans: Colors of the points should be written here (1D np.array of np.uint32).
        """
        pass

    def to_object(self) -> object:
        """Produces the same kwargs as taken by __init__. Must be implemented."""
        pass


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

    def paint(self, points, ans):
        ans[:] = sierpinski_numba(points, self.depth)


# All supported painters.
ALL_PAINTERS = [MandelbroidPainter, MadnelbrotHighPrecisionPainter, JuliaPainter, SierpinskiCarpetPainter]
PAINTERS_INDEX = {ALL_PAINTERS[i].__name__: i for i in range(len(ALL_PAINTERS))}
