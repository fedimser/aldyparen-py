from .julia import JuliaPainter
from .mandelbroid import MandelbroidPainter
from .mandelbrot_hp import MandelbrotHighPrecisionPainter
from .sierpinski import SierpinskiCarpetPainter


class Painter:
    """Abstract class for all painters.

    __init__ must take all args as kwargs. It must create default instance with no args.
    """

    @staticmethod
    def deserialize(class_name: str, data: dict) -> 'Painter':
        assert class_name in PAINTERS_INDEX
        painter_class = ALL_PAINTERS[PAINTERS_INDEX[class_name]]
        return painter_class(**data)

    def paint(self, points: 'np.ndarray', ans: 'np.ndarray') -> 'np.ndarray':
        """Paints given points. Must be implemented (unless this is a high-precision painter).

        May set `self.warning` to a message string if there was some problem.

        :param points: Points to be painted (1D np.array of np.complex128).
        :param ans: Colors of the points should be written here (1D np.array of np.uint32).
        """
        pass

    def to_object(self) -> object:
        """Produces the same kwargs as taken by __init__. Must be implemented."""
        pass


# All supported painters.
ALL_PAINTERS = [MandelbroidPainter, MandelbrotHighPrecisionPainter, JuliaPainter,
                SierpinskiCarpetPainter]
PAINTERS_INDEX = {ALL_PAINTERS[i].__name__: i for i in range(len(ALL_PAINTERS))}
