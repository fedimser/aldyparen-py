import warnings

import numba
import numpy as np

from aldyparen.util import prepare_function


class MandelbroidPainter:
    def __init__(self, gen_function="z*z+c", max_iter=100, radius=2):
        assert 1 <= max_iter <= 1000000, "bad max_iter"
        self.gen_function = gen_function
        self.max_iter = max_iter
        self.radius = radius
        self.gen_function_prepared = prepare_function(gen_function, variables=['c', 'z'])
        self.paint_func = None

    def to_object(self):
        return {"gen_function": self.gen_function,
                "radius": self.radius,
                "max_iter": self.max_iter}

    def paint(self, points, ans):
        if self.paint_func is None:
            numba_namespace = {"numba": numba, "np": np}
            source = "\n".join([
                f'@numba.vectorize("u4(c16)", target="parallel")',
                f'def painter__(c):',
                f'  z = 0',
                f'  for i in range({self.max_iter}):',
                f'    z = {self.gen_function_prepared}',
                f'    if np.abs(z) > {self.radius}: return i',
                f'  return {self.max_iter}',
            ])
            exec(source, numba_namespace)
            self.paint_func = numba_namespace["painter__"]

        warnings.filterwarnings("ignore", message="overflow")
        try:
            ans[:] = self.paint_func(points)
        except Exception as e:
            self.warning = f"Error in mandelbroid painter: {e}"
            ans[:] = np.zeros_like(points, dtype=np.uint32)
