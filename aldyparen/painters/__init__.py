from .base import HighPrecisionPainter, Painter
from .julia import JuliaPainter
from .mandelbroid import MandelbroidPainter
from .mandelbroid_hp import MandelbroidHighPrecisionPainter
from .mandelbrot_hp import MandelbrotHighPrecisionPainter
from .sierpinski import SierpinskiCarpetPainter

# All supported painters.
ALL_PAINTERS = [
    MandelbroidPainter,
    MandelbroidHighPrecisionPainter,
    MandelbrotHighPrecisionPainter,
    JuliaPainter,
    SierpinskiCarpetPainter,
]
PAINTERS_INDEX = {ALL_PAINTERS[i].__name__: i for i in range(len(ALL_PAINTERS))}
