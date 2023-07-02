import os

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt

from aldyparen.graphics import Frame, StaticRenderer, InteractiveRenderer, Transform, ColorPalette
from aldyparen.painters import MandelbroidPainter, SierpinskiCarpetPainter

GOLDEN_DIR = f"{os.getcwd()}/goldens"

PALETTE = ColorPalette.gradient('yellow', 'black', size=21)

def assert_picture(picture, golden_name, overwrite=False):
    golden_path = f"{GOLDEN_DIR}/{golden_name}.bmp"
    if overwrite:
        plt.imsave(golden_path, picture)
    golden = None
    if os.path.exists(golden_path):
        golden = mpimg.imread(golden_path)
    if golden is None or picture.shape != golden.shape  or not np.allclose(picture, golden):
        plt.imsave(f"{GOLDEN_DIR}/{golden_name}_expected.bmp", picture)
        raise AssertionError(f"Golden mismatch: {golden_name}")


def test_renders_mandelbroids():
    renderer = StaticRenderer(200, 200)
    transform = Transform(0, 4, 0.0)
    for gen_function in ["z*z+c", "z*z*z+c", "z*z*z+3**z+c"]:
        frame = Frame(MandelbroidPainter(gen_function=gen_function, max_iter=20), transform, PALETTE)
        assert_picture(renderer.render(frame), f"mandelbroid_{gen_function}")

def test_renders_sierpinski_carpet():
    renderer = StaticRenderer(200, 200)
    transform = Transform(0.5 + 0.5j, 1, 0.0)
    palette = ColorPalette.gradient('black', 'white', size=2)
    frame = Frame(SierpinskiCarpetPainter(depth=4), transform, palette)
    assert_picture(renderer.render(frame), "sierpinski_carpet")

# TODO: add test for cascade rendering with InteractiveRenderer.
