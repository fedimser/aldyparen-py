from typing import Any

from aldyparen import Transform
from aldyparen.graphics import ColorPalette
from aldyparen.painters import (
    MandelbroidHighPrecisionPainter,
    MandelbroidPainter,
    MandelbrotHighPrecisionPainter,
    Painter,
)

BS_PALETTE = ColorPalette.categorical(["black"]) + ColorPalette.gradient("orange", "blue", 20)

PRESET_NAMES = ["mandelbrot", "mandelbrot_hp", "burning_ship", "burning_ship_hp"]


def load_preset(name: str) -> tuple[Painter, Transform, ColorPalette]:
    palette = BS_PALETTE
    match name:
        case "mandelbrot":
            painter = MandelbroidPainter(gen_function="z*z+c", max_iter=100, radius=2)
            transform = Transform.create(scale=4)
        case "mandelbrot_hp":
            painter = MandelbrotHighPrecisionPainter(max_iter=100)
            transform = Transform.create(scale=4)
        case "burning_ship":
            painter = (MandelbroidPainter(gen_function="(abs(real(z))+1j*abs(imag(z)))**2+c"),)
            transform = Transform.create(center=-1.769 - 0.035j, scale_log10=-0.8, rotation_deg=180)
        case "burning_ship_hp":
            painter = MandelbroidHighPrecisionPainter(gen_function="abscw(z)**2+c", max_iter=100, precision=4)
            transform = Transform.create(center=-1.769 - 0.035j, scale_log10=-0.8, rotation_deg=180)
    return painter, transform, palette
