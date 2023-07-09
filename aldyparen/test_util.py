import os

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt

GOLDEN_DIR = os.path.join(os.getcwd(), "goldens")


def _assert_picture(picture, golden_name, overwrite=False):
    golden_path = os.path.join(GOLDEN_DIR, golden_name + ".bmp")
    if overwrite:
        plt.imsave(golden_path, picture)
    golden = None
    if os.path.exists(golden_path):
        golden = mpimg.imread(golden_path)
    if golden is None or picture.shape != golden.shape or not np.array_equal(picture, golden):
        plt.imsave(os.path.join(GOLDEN_DIR, golden_name + "_expected.bmp"), picture)
        raise AssertionError(f"Golden mismatch: {golden_name}")
