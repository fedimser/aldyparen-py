from aldyparen import Transform
from aldyparen.painters import MandelbroidPainter, MandelbrotHighPrecisionPainter, MandelbroidHighPrecisionPainter
from aldyparen.graphics import ColorPalette
from typing import Any

BS_PALETTE = ColorPalette.categorical(["black"]) + ColorPalette.gradient("orange", "blue", 20)

PRESETS = {
    "mandelbrot": (
        MandelbroidPainter(gen_function="z*z+c", max_iter=100, radius=2),
        Transform.create(scale=4),
        BS_PALETTE),
    "mandelbrot_hp": (
        MandelbrotHighPrecisionPainter(max_iter=100, ),
        Transform.create(scale=4),
        BS_PALETTE),
    "burning_ship": (MandelbroidPainter(gen_function="(abs(real(z))+1j*abs(imag(z)))**2+c"),
                     Transform.create(center=-1.769 - 0.035j, scale_log10=-0.8, rotation_deg=180),
                     BS_PALETTE),
    "burning_ship_hp": (
        MandelbroidHighPrecisionPainter(gen_function="abscw(z)**2+c",
                                        max_iter=100,
                                        precision=4),
        Transform.create(center=-1.769 - 0.035j, scale_log10=-0.8, rotation_deg=180),
        BS_PALETTE),
}  # type: dict[str, tuple[Any, Transform, ColorPalette]]
