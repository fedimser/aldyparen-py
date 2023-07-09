import json
import os

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt

from aldyparen.graphics import Frame, StaticRenderer, Transform, ColorPalette
from aldyparen.painters import MandelbroidPainter, SierpinskiCarpetPainter, MandelbrotHighPrecisionPainter, \
    JuliaPainter, ALL_PAINTERS
from aldyparen.util import SUPPORTED_FUNCTIONS

GOLDEN_DIR = os.path.join(os.getcwd(), "goldens")


def _assert_picture(picture, golden_name, overwrite=False):
    golden_path = os.path.join(GOLDEN_DIR, golden_name + ".bmp")
    if overwrite:
        plt.imsave(golden_path, picture)
    golden = None
    if os.path.exists(golden_path):
        golden = mpimg.imread(golden_path)
    if golden is None or picture.shape != golden.shape or not np.allclose(picture, golden):
        plt.imsave(os.path.join(GOLDEN_DIR, golden_name + "_expected.bmp"), picture)
        raise AssertionError(f"Golden mismatch: {golden_name}")


def test_defaults():
    for painter_class in ALL_PAINTERS:
        painter = painter_class()
        config1 = painter.to_object()
        config1_json = json.dumps(config1)
        assert config1_json == json.dumps(json.loads(config1_json))
        painter2 = painter_class(**config1)
        config2 = painter2.to_object()
        assert config1_json == json.dumps(config2)


def test_mandelbroid_rejects_bad_functions():
    error_msg = None
    try:
        MandelbroidPainter(gen_function="f(c,z)")
    except ValueError as err:
        error_msg = str(err)
    assert "Unexpected token: f" in error_msg


def test_mandelbroid_supports_all_functions():
    gen_function = '+'.join(func + '(c+z+1j)' for func in SUPPORTED_FUNCTIONS)
    MandelbroidPainter(gen_function=gen_function)


def test_renders_mandelbroids():
    palette = ColorPalette.gradient('yellow', 'black', size=21)
    renderer = StaticRenderer(200, 200)
    transform = Transform.create(scale=4)
    funcs = ["z*z+c", "z*z*z+c", "z*z*z+3**z+c"]
    for i in range(3):
        frame = Frame(MandelbroidPainter(gen_function=funcs[i], max_iter=20), transform, palette)
        _assert_picture(renderer.render(frame), f"mandelbroid_{i}")


def test_renders_mandelbrot_high_precision():
    renderer = StaticRenderer(100, 100)
    palette = ColorPalette.gradient('white', 'black', size=10)
    p = MandelbrotHighPrecisionPainter(max_iter=100)
    frame1 = Frame(p, Transform.create(scale=4), palette)
    _assert_picture(renderer.render(frame1), f"mandelbrot_hp")
    center = np.complex128(-1.99977406013629035931 - 0.00000000329004032147j)
    tr2 = Transform.create(center=center, scale_log10=-6, rotation=1.0)
    frame2 = Frame(p, tr2, palette)
    _assert_picture(renderer.render(frame2), f"mandelbrot_hp_zoom")


def test_renders_julia_set():
    renderer = StaticRenderer(200, 200)
    transform = Transform.create(center=0.1+0.2j, scale=3, rotation=0.8)
    # Newton fractal for P(z)=z^3-1.
    frame = Frame(JuliaPainter(func="z-(z**3-1)/(3*z**2)"), transform, ColorPalette.default())
    _assert_picture(renderer.render(frame), "newton_z3m1")

    # Newton fractal for P(z)=(z-1)(z-2)(z-3).
    frame = Frame(JuliaPainter(func="z-(z**3-6*z**2+11*z-6)/(3*z**2-12*z+11)"), Transform.create(center=2, scale=5),
                  ColorPalette.default())
    _assert_picture(renderer.render(frame), "newton_poly3")


def test_renders_sierpinski_carpet():
    renderer = StaticRenderer(200, 200)
    transform = Transform.create(center=0.5 + 0.5j)
    palette = ColorPalette.gradient('black', 'white', size=2)
    frame = Frame(SierpinskiCarpetPainter(depth=4), transform, palette)
    _assert_picture(renderer.render(frame), "sierpinski_carpet")


def _verify_serialization(frame1: Frame):
    data1 = frame1.serialize()
    frame2 = Frame.deserialize(data1)
    data2 = frame2.serialize()
    assert data1 == data2
    assert frame1.transform == frame2.transform
    assert frame1.palette == frame2.palette
    assert frame1.painter.__class__ == frame2.painter.__class__
    assert frame1.painter.to_object() == frame2.painter.to_object()
    pic1 = StaticRenderer(10, 10).render(frame1)
    pic2 = StaticRenderer(10, 10).render(frame2)
    assert np.array_equal(pic1, pic2)


def test_serialization():
    _verify_serialization(Frame(SierpinskiCarpetPainter(depth=4), Transform.create(), ColorPalette.default()))
    _verify_serialization(
        Frame(MandelbroidPainter(gen_function="z**3+sin(z)+c"), Transform.create(center=2 + 3j, scale=10, rotation=3.1),
              ColorPalette.random()))
    _verify_serialization(
        Frame(MandelbrotHighPrecisionPainter(), Transform.create(center=2j, scale=1e-3, rotation=-6),
              ColorPalette.grayscale(20)))
    _verify_serialization(
        Frame(JuliaPainter(func="z-(z**2-1)/(2*z)", iters=10), Transform.create(scale=5), ColorPalette.random(3)))
