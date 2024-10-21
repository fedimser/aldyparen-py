import json
import random

import numpy as np
import pytest

from aldyparen.graphics import Frame, StaticRenderer, Transform, ColorPalette
from aldyparen.gui.presets import PRESETS
from aldyparen.painters import MandelbroidPainter, SierpinskiCarpetPainter, \
    MandelbrotHighPrecisionPainter, \
    JuliaPainter, ALL_PAINTERS, MandelbroidHighPrecisionPainter
from aldyparen.test_util import _assert_picture
from aldyparen.util import SUPPORTED_FUNCTIONS


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
    p1 = MandelbrotHighPrecisionPainter(max_iter=100)
    p2 = MandelbroidHighPrecisionPainter(gen_function="z*z+c", max_iter=100, radius=2)
    frame1 = Frame(p1, Transform.create(scale=4), palette)
    _assert_picture(renderer.render(frame1), f"mandelbrot_hp")
    frame2 = Frame(p2, Transform.create(scale=4), palette)
    _assert_picture(renderer.render(frame2), f"mandelbrot_hp")

    center = np.complex128(-1.99977406013629035931 - 0.00000000329004032147j)
    tr2 = Transform.create(center=center, scale_log10=-6, rotation=1.0)
    frame1 = Frame(p1, tr2, palette)
    _assert_picture(renderer.render(frame1), f"mandelbrot_hp_zoom")
    frame2 = Frame(p2, tr2, palette)
    _assert_picture(renderer.render(frame2), f"mandelbrot_hp_zoom")


def test_renders_julia_set():
    renderer = StaticRenderer(200, 200)
    transform = Transform.create(center=0.1 + 0.2j, scale=3, rotation=0.8)
    # Newton fractal for P(z)=z^3-1.
    frame = Frame(JuliaPainter(func="z-(z**3-1)/(3*z**2)"), transform, ColorPalette.default())
    _assert_picture(renderer.render(frame), "newton_z3m1")

    # Newton fractal for P(z)=(z-1)(z-2)(z-3).
    frame = Frame(JuliaPainter(func="z-(z**3-6*z**2+11*z-6)/(3*z**2-12*z+11)"),
                  Transform.create(center=2, scale=5),
                  ColorPalette.default())
    _assert_picture(renderer.render(frame), "newton_poly3")


def test_renders_sierpinski_carpet():
    renderer = StaticRenderer(200, 200)
    transform = Transform.create(center=0.5 + 0.5j)
    palette = ColorPalette.gradient('black', 'white', size=2)
    frame = Frame(SierpinskiCarpetPainter(depth=4), transform, palette)
    _assert_picture(renderer.render(frame), "sierpinski_carpet")


@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_renders_presets(preset_name):
    renderer = StaticRenderer(256, 256)
    painter, transform, palette = PRESETS[preset_name]
    frame = Frame(painter, transform, palette)
    _assert_picture(renderer.render(frame), "preset_" + preset_name, max_mismatched_pixels=20)


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
    _verify_serialization(
        Frame(SierpinskiCarpetPainter(depth=4), Transform.create(), ColorPalette.default()))
    _verify_serialization(
        Frame(MandelbroidPainter(gen_function="z**3+sin(z)+c"),
              Transform.create(center=2 + 3j, scale=10, rotation=3.1),
              ColorPalette.random()))
    _verify_serialization(
        Frame(MandelbrotHighPrecisionPainter(),
              Transform.create(center=2j, scale=1e-3, rotation=-6),
              ColorPalette.grayscale(20)))
    _verify_serialization(
        Frame(JuliaPainter(func="z-(z**2-1)/(2*z)", iters=10), Transform.create(scale=5),
              ColorPalette.random(3)))
