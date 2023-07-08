import time

import numpy as np

from aldyparen.graphics import Frame, StaticRenderer, Transform, InteractiveRenderer, ChunkingRenderer, ColorPalette
from aldyparen.painters import MandelbroidPainter


def test_transform_to_string():
    transform = Transform(np.complex128(1 + 0.05j), 1e-3, np.pi / 4)
    assert str(transform) == "c=(1.00000e+00 5.00000e-02) s=1.00e-03 r=45.0°"


def test_transform_manipulation():
    t = Transform.create(0, 1, 0)
    t = t.translate(1.5, 2.5)
    assert t == Transform.create(-1.5 - 2.5j, 1, 0)
    t = t.rotate_at_point(1 + 2j, 0.5)
    assert t == Transform.create(-2.5812685153180333 - 2.2654093376150515j, 1, 0.5)
    t = t.scale_at_point(-1 - 3j, 3)
    assert t == Transform.create(-8.86519365379857 - 0.5718815642945119j, 3, 0.5)


def test_transform_creation():
    t = Transform.create(center=-1.76 - 0.03j, scale=0.185)
    assert t == Transform(np.complex128(-1.76 - 0.03j), 0.185, 0)
    assert t.rotate_at_frame_center(np.pi) == Transform(np.complex128(1.76 + 0.03j), 0.185, np.pi)


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
    palette = ColorPalette.gradient('yellow', 'black', size=21)
    frame = Frame(MandelbroidPainter(gen_function="z*z+c", max_iter=10), transform, palette)

    pic1 = renderer1.render(frame)
    pic2 = renderer2.render(frame)
    pic3 = renderer3.render(frame)
    pic4 = _render_with_interactive_renderer(w, h, frame)

    assert pic1.shape == (h, w, 3)
    assert np.sum(np.abs(pic1 - pic2)) == 0
    assert np.sum(np.abs(pic1 - pic3)) == 0
    assert np.sum(np.abs(pic1 - pic4)) == 0
