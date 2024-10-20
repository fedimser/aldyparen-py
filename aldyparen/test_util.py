import os

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt

GOLDEN_DIR = os.path.join(os.getcwd(), "goldens")


def _match_pictures(x: np.ndarray, y: np.ndarray, max_mismatched_pixels: int):
    if x is None or y is None or x.shape != y.shape:
        return False
    mismatch_count = np.sum(x != y)
    if mismatch_count > max_mismatched_pixels:
        print(f"Mismatched pixels: {mismatch_count}")
        return False
    return True


def _assert_picture(picture, golden_name, overwrite=False, max_mismatched_pixels=0):
    golden_path = os.path.join(GOLDEN_DIR, golden_name + ".bmp")
    if overwrite:
        plt.imsave(golden_path, picture)
    golden = None
    if os.path.exists(golden_path):
        golden = mpimg.imread(golden_path)
    if not _match_pictures(picture, golden, max_mismatched_pixels):
        plt.imsave(os.path.join(GOLDEN_DIR, golden_name + "_expected.bmp"), picture)
        raise AssertionError(f"Golden mismatch: {golden_name}")
