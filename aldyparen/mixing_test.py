from aldyparen.mixing import mix_functions, make_animation
from aldyparen.painters import MandelbroidPainter
from aldyparen.graphics import Frame, Transform, ColorPalette
import numpy as np


def test_mix_functions():
    f1 = "2*x+3*y-5"
    f2 = "1*x+4*y-10"
    f3 = mix_functions(f1, f2, 0.25)
    assert (f3 == '1.75*x+3.25*y-6.25')


def test_make_animation():
    p1 = MandelbroidPainter(gen_function="z**2+c", max_iter=50, radius=10)
    p2 = MandelbroidPainter(gen_function="z**10+c", max_iter=100, radius=20)
    t1 = Transform(0, 1, 0)
    t2 = Transform(2 + 4j, 25, 2)
    palette1 = ColorPalette(colors=np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=np.uint8))
    palette2 = ColorPalette(colors=np.ones((5, 3), dtype=np.uint8) * 10)
    frame1 = Frame(p1, t1, palette1)
    frame2 = Frame(p2, t2, palette2)

    animation = make_animation(frame1, frame2, 10)
    assert len(animation) == 11
    assert animation[0] == frame1
    assert animation[10] == frame2
    mid_frame = animation[5]
    assert mid_frame.transform == Transform(1 + 2j, 5, 1)
    mid_painter = mid_frame.painter  # type: MandelbroidPainter
    assert mid_painter.max_iter == 75
    assert np.allclose(mid_painter.radius, 15)
    assert mid_painter.gen_function == "z**6.0+c"
    mid_colors = mid_frame.palette.colors
    assert mid_colors.shape == (5, 3)
    np.testing.assert_equal(mid_colors, [[6, 7, 8], [9, 10, 11], [12, 13, 14], [6, 7, 8], [9, 10, 11]])
