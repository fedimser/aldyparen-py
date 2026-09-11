import functools

import numpy as np

from ..graphics import ColorPalette, Transform
from ..painters import (
    LyapunovFractalPainter,
    MagneticPendulumPainter,
    MandelbroidHighPrecisionPainter,
    MandelbroidPainter,
    MandelbrotHighPrecisionPainter,
    Painter,
)

PRESET_NAMES = ["mandelbrot", "mandelbrot_hp", "burning_ship", "burning_ship_hp", "lyapunov", "magnetic_pendulum"]


def _default_palette() -> ColorPalette:
    return ColorPalette.categorical(["black"]) + ColorPalette.gradient("orange", "blue", 20)


def _lyapunov_palette() -> ColorPalette:
    colors = np.zeros((65, 3), dtype=np.uint8)
    colors[1::2] = ColorPalette.gradient("lightcyan", "navy", 32).colors
    colors[2::2] = ColorPalette.gradient("yellow", "darkred", 32).colors
    return ColorPalette(colors)


def _magnetic_pendulum_palette() -> ColorPalette:
    colors = np.zeros((49, 3), dtype=np.uint8)
    magnet_colors = np.array([[255, 80, 70], [70, 150, 255], [255, 220, 60]], dtype=np.float64)
    for time_bin in range(16):
        brightness = 1.0 - 0.65 * time_bin / 15
        colors[1 + 3 * time_bin : 4 + 3 * time_bin] = np.round(magnet_colors * brightness)
    return ColorPalette(colors)


@functools.cache
def load_preset(name: str) -> tuple[Painter, Transform, ColorPalette]:
    palette = _default_palette()
    match name:
        case "mandelbrot":
            painter = MandelbroidPainter(gen_function="z*z+c", max_iter=100, radius=2)
            transform = Transform.create(scale=4)
        case "mandelbrot_hp":
            painter = MandelbrotHighPrecisionPainter(max_iter=100)
            transform = Transform.create(scale=4)
        case "burning_ship":
            painter = MandelbroidPainter(gen_function="(abs(real(z))+1j*abs(imag(z)))**2+c")
            transform = Transform.create(center=-1.769 - 0.035j, scale_log10=-0.8, rotation_deg=180)
        case "burning_ship_hp":
            painter = MandelbroidHighPrecisionPainter(gen_function="abscw(z)**2+c", max_iter=100, precision=4)
            transform = Transform.create(center=-1.769 - 0.035j, scale_log10=-0.8, rotation_deg=180)
        case "lyapunov":
            painter = LyapunovFractalPainter(sequence="AABAB", warmup=100, iterations=100, color_scale=10)
            transform = Transform.create(center=3 + 3j, scale=2)
            palette = _lyapunov_palette()
        case "magnetic_pendulum":
            painter = MagneticPendulumPainter()
            transform = Transform.create(scale=4)
            palette = _magnetic_pendulum_palette()
        case _:
            raise ValueError(f"Unknown preset name: {name}")
    return painter, transform, palette
