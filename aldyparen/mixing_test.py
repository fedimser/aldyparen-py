import numpy as np
import pytest

from aldyparen.graphics import ColorPalette, Frame, Transform
from aldyparen.mixing import make_animation, mix_functions, mix_painters, mix_palettes
from aldyparen.painters import (
    JuliaPainter,
    MandelbroidPainter,
    MandelbrotHighPrecisionPainter,
    SierpinskiCarpetPainter,
)


def test_mix_functions():
    f1 = "2*x+3*y-5"
    f2 = "1*x+4*y-10"
    f3 = mix_functions(f1, f2, 0.25)
    assert f3 == "1.75*x+3.25*y-6.25"


def test_mix_functions_fractional():
    f1 = "2*x+3*x**2"
    f2 = "2.5*x+3.5*x**2.0"
    f3 = mix_functions(f1, f2, 0.5)
    assert f3 == "2.25*x+3.25*x**2.0"


def test_mix_functions_scientific_notation():
    assert mix_functions("z+0.0000000001", "z+0.0000000003", 0.5) == "z+0.0000000002"


@pytest.mark.parametrize(
    ("f1", "f2"),
    [
        ("z+1", "z+1+2"),
        ("z+1", "z+x"),
        ("z+1", "c+2"),
    ],
)
def test_mix_functions_rejects_incompatible_expressions(f1, f2):
    with pytest.raises(ValueError):
        mix_functions(f1, f2, 0.5)


def test_mix_palettes_extends_second_palette():
    palette1 = ColorPalette(np.array([[0, 10, 20], [20, 30, 40]], dtype=np.uint8))
    palette2 = ColorPalette(np.array([[100, 110, 120]], dtype=np.uint8))

    mixed = mix_palettes(palette1, palette2, 0.5)

    np.testing.assert_array_equal(mixed.colors, [[50, 60, 70], [60, 70, 80]])


def test_mix_supported_and_unsupported_painters():
    julia = mix_painters(
        JuliaPainter(func="z+1", iters=10, tolerance=0.01, max_colors=2),
        JuliaPainter(func="z+3", iters=20, tolerance=0.03, max_colors=4),
        0.5,
    )
    assert julia.to_object() == {"func": "z+2.0", "iters": 15, "tolerance": 0.02, "max_colors": 3}

    mandelbrot = mix_painters(MandelbrotHighPrecisionPainter(10), MandelbrotHighPrecisionPainter(20), 0.5)
    assert mandelbrot.max_iter == 15

    painter = SierpinskiCarpetPainter(depth=2)
    assert mix_painters(painter, painter, 0.5) is painter
    with pytest.raises(ValueError, match="Cannot mix painters"):
        mix_painters(painter, SierpinskiCarpetPainter(depth=3), 0.5)


def test_make_animation():
    p1 = MandelbroidPainter(gen_function="z**2+c", max_iter=50, radius=10)
    p2 = MandelbroidPainter(gen_function="z**10+c", max_iter=100, radius=20)
    t1 = Transform.create(scale=1)
    t2 = Transform.create(center=2 + 4j, scale=25, rotation=2)
    palette1 = ColorPalette(colors=np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=np.uint8))
    palette2 = ColorPalette(colors=np.ones((5, 3), dtype=np.uint8) * 10)
    frame1 = Frame(p1, t1, palette1)
    frame2 = Frame(p2, t2, palette2)

    animation = make_animation(frame1, frame2, 10)
    assert len(animation) == 11
    assert animation[0] == frame1
    assert animation[10] == frame2
    mid_frame = animation[5]
    assert mid_frame.transform == Transform.create(center=1 + 2j, scale=5, rotation=1)
    mid_painter = mid_frame.painter  # type: MandelbroidPainter
    assert mid_painter.max_iter == 75
    assert np.allclose(mid_painter.radius, 15)
    assert mid_painter.gen_function == "z**6.0+c"
    mid_colors = mid_frame.palette.colors
    assert mid_colors.shape == (5, 3)
    np.testing.assert_equal(mid_colors, [[6, 7, 8], [9, 10, 11], [12, 13, 14], [6, 7, 8], [9, 10, 11]])
