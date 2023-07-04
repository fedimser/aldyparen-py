import os

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt

from aldyparen.graphics import Frame, StaticRenderer, Transform, ColorPalette
from aldyparen.painters import MandelbroidPainter, SierpinskiCarpetPainter, MadnelbrotHighPrecisionPainter

GOLDEN_DIR = os.path.join(os.getcwd(), "goldens")

PALETTE = ColorPalette.gradient('yellow', 'black', size=21)


def assert_picture(picture, golden_name, overwrite=False):
    golden_path = os.path.join(GOLDEN_DIR, golden_name + ".bmp")
    if overwrite:
        plt.imsave(golden_path, picture)
    golden = None
    if os.path.exists(golden_path):
        golden = mpimg.imread(golden_path)
    if golden is None or picture.shape != golden.shape or not np.allclose(picture, golden):
        plt.imsave(os.path.join(GOLDEN_DIR, golden_name + "_expected.bmp"), picture)
        raise AssertionError(f"Golden mismatch: {golden_name}")


def test_renders_mandelbroids():
    renderer = StaticRenderer(200, 200)
    transform = Transform(0, 4, 0.0)
    funcs = ["z*z+c", "z*z*z+c", "z*z*z+3**z+c"]
    for i in range(3):
        frame = Frame(MandelbroidPainter(gen_function=funcs[i], max_iter=20), transform, PALETTE)
        assert_picture(renderer.render(frame), f"mandelbroid_{i}")


def test_renders_sierpinski_carpet():
    renderer = StaticRenderer(200, 200)
    transform = Transform(np.complex128(0.5 + 0.5j), 1, 0.0)
    palette = ColorPalette.gradient('black', 'white', size=2)
    frame = Frame(SierpinskiCarpetPainter(depth=4), transform, palette)
    assert_picture(renderer.render(frame), "sierpinski_carpet")


def _verify_serialization(frame1: Frame):
    data1 = frame1.serialize()
    frame2 = Frame.deserialize(data1)
    data2 = frame2.serialize()
    assert data1 == data2
    assert frame1.transform == frame2.transform
    assert frame1.palette == frame2.palette
    assert frame1.painter.__class__ == frame2.painter.__class__
    assert frame1.painter.to_object() == frame2.painter.to_object()


def test_serialization():
    _verify_serialization(Frame(SierpinskiCarpetPainter(depth=4), Transform(0, 1, 0), ColorPalette.default()))
    _verify_serialization(
        Frame(MandelbroidPainter(gen_function="z**3+sin(z)+c"), Transform(2 + 3j, 10, 3.1), ColorPalette.random()))
    _verify_serialization(
        Frame(MadnelbrotHighPrecisionPainter(), Transform(2j, 1e-3, -6), ColorPalette.grayscale(20)))


def test_transform_to_string():
    transform = Transform(np.complex128(1 + 0.05j), 1e-3, np.pi / 4)
    print(str(transform))
    assert str(transform) == "c=(1.00000e+00 5.00000e-02) s=1.00e-03 r=45.0°"

# TODO: add test for cascade rendering with InteractiveRenderer.
