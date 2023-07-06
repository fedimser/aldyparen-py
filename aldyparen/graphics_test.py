import os
import time

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt

from aldyparen.graphics import Frame, StaticRenderer, Transform, ColorPalette, InteractiveRenderer, ChunkingRenderer
from aldyparen.painters import MandelbroidPainter, SierpinskiCarpetPainter, MandelbrotHighPrecisionPainter, JuliaPainter

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


def test_renders_mandelbrot_high_precision():
    renderer = StaticRenderer(100, 100)
    palette = ColorPalette.gradient('white', 'black', size=10)
    p = MandelbrotHighPrecisionPainter(max_iter=100)
    frame1 = Frame(p, Transform(0, 4, 0), palette)
    assert_picture(renderer.render(frame1), f"mandelbrot_hp")
    center = np.complex128(-1.99977406013629035931 - 0.00000000329004032147j)
    frame2 = Frame(p, Transform(center, 1e-6, 0.0), palette)
    assert_picture(renderer.render(frame2), f"mandelbrot_hp_zoom")


def test_renders_julia_set():
    renderer = StaticRenderer(200, 200)
    transform = Transform(np.complex128(0), 3, 0.0)
    # Newton fractal for P(z)=z^3-1.
    frame = Frame(JuliaPainter(func="z-(z**3-1)/(3*z**2)"), transform, ColorPalette.default())
    assert_picture(renderer.render(frame), "newton_z3m1")

    # Newton fractal for P(z)=(z-1)(z-2)(z-3).
    frame = Frame(JuliaPainter(func="z-(z**3-6*z**2+11*z-6)/(3*z**2-12*z+11)"), Transform(2, 5, 0),
                  ColorPalette.default())
    assert_picture(renderer.render(frame), "newton_poly3")


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
        Frame(MandelbrotHighPrecisionPainter(), Transform(2j, 1e-3, -6), ColorPalette.grayscale(20)))


def test_transform_to_string():
    transform = Transform(np.complex128(1 + 0.05j), 1e-3, np.pi / 4)
    assert str(transform) == "c=(1.00000e+00 5.00000e-02) s=1.00e-03 r=45.0°"


def _render_with_interactive_renderer(w: int, h: int, frame: Frame) -> np.ndarray:
    results = []
    r = InteractiveRenderer(w, h, lambda pic: results.append(pic), downsample_factor=3)
    r.render_async(frame)
    while len(results) == 0 or not r.renderer_thread.is_idle:
        time.sleep(0.01)
        r.tick()
    r.halt()
    r.renderer_thread.quit()
    return results[-1]


def test_different_renderers():
    w, h = 53, 31
    renderer1 = StaticRenderer(w, h)
    renderer2 = ChunkingRenderer(w, h, chunk_size=17)
    renderer3 = ChunkingRenderer(w, h, chunk_size=100000)
    transform = Transform(0, 4, 0.0)
    frame = Frame(MandelbroidPainter(gen_function="z*z+c", max_iter=10), transform, PALETTE)

    pic1 = renderer1.render(frame)
    pic2 = renderer2.render(frame)
    pic3 = renderer3.render(frame)
    pic4 = _render_with_interactive_renderer(w, h, frame)

    assert pic1.shape == (h, w, 3)
    assert np.sum(np.abs(pic1 - pic2)) == 0
    assert np.sum(np.abs(pic1 - pic3)) == 0
    assert np.sum(np.abs(pic1 - pic4)) == 0
