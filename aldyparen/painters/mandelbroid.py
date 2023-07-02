import warnings

import numba
import numpy as np
import re

from aldyparen.util import prepare_function


class MandelbroidPainter:
    def __init__(self, gen_function="z*z+c", max_iter=100, radius=2):
        self.gen_function = gen_function
        self.max_iter = max_iter
        self.radius = radius
        gen_function_prepared = prepare_function(gen_function, variables=['c', 'z'])

        numba_namespace = {"numba": numba, "np": np}
        source = "\n".join([
            f'@numba.vectorize("u4(c16)", target="parallel")',
            f'def painter__(c):',
            f'  z = 0',
            f'  for i in range({max_iter}):',
            f'    z = {gen_function_prepared}',
            f'    if np.abs(z) > {radius}: return i',
            f'  return {max_iter}',
        ])
        exec(source, numba_namespace)
        self.paint_func = numba_namespace["painter__"]

    def to_object(self):
        return {"gen_function": self.gen_function,
                "radius": self.radius,
                "max_iter": self.max_iter}

    def paint(self, points):
        warnings.filterwarnings("ignore", message="overflow")
        try:
            return self.paint_func(points)
        except Exception as e:
            print(f"Error in mandelbroid painter: {e}")
            return np.zeros_like(points, dtype=np.uint32)
