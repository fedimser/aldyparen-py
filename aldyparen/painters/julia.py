import numpy as np
import numba
import warnings

from aldyparen.util import prepare_function


class JuliaPainter:
    """Paints complex plane based on where they get by applying `func` many times.

    Points z1 and z2 will be painted the same color if `abs(F(z1),F(z2))<tolerance`, where F(z)=f(f(f...(z)..))`, where
    `f` is applied `iters` times.
    If `abs(F(z)) > 1/tolerance`, we assume that z is sent to infinity by this process - all those points will be
    painted the same color.

    This can be used to draw [Newton fractal](https://en.wikipedia.org/wiki/Newton_fractal). To get Newton fractal for
    function P, use `JuliaPainter` with `f(z)=z-P(z)/P'(z)`.
    """

    def __init__(self, func="z-(z*z*z-1)/(3*z*z)", iters=100, tolerance=1e-9, max_colors=20):
        self.func = func
        self.iters = iters
        assert 0 < tolerance < 1e-5
        self.tolerance = tolerance
        self.func_prepared = prepare_function(func, variables=['z', 'c'])
        self.max_colors = max_colors
        self.iterate_func = None
        self.attractors = [np.inf]
        self.warning = None

    def to_object(self):
        return {"func": self.func,
                "iters": self.iters,
                "tolerance": self.tolerance,
                "max_colors": self.max_colors}

    def paint(self, points: np.ndarray, ans: np.ndarray):
        if self.iterate_func is None:
            numba_namespace = {"numba": numba, "np": np}
            source = "\n".join([
                f'@numba.vectorize("c16(c16)", target="parallel")',
                f'def _iterate(z):',
                f'  for i in range({self.iters}):',
                f'    z = {self.func_prepared}',
                f'  return z',
            ])
            exec(source, numba_namespace)
            self.iterate_func = numba_namespace["_iterate"]

        warnings.filterwarnings("ignore", message="overflow")
        try:
            points_after = self.iterate_func(points)
        except Exception as e:
            print(f"Error in _iterate: {e}")
            ans[:] = np.zeros_like(points, dtype=np.uint32)
            return

        n = len(points_after)
        assert ans.shape == (n,)

        color_limit_exceeded = False
        for i in range(n):
            z = points_after[i]
            if np.isnan(z) or np.isinf(z) or np.abs(z) > (1 / self.tolerance):
                ans[i] = 0
                continue
            idx = -1
            for j in range(len(self.attractors)):
                if np.abs(self.attractors[j] - z) < self.tolerance:
                    idx = j
                    break
            if idx == -1:
                if len(self.attractors) < self.max_colors:
                    idx = len(self.attractors)
                    self.attractors.append(z)
                else:
                    color_limit_exceeded = True
                    idx = 0
            ans[i] = idx

        if color_limit_exceeded:
            self.warning = "Warning! Color limit exceeded, extra colors were mapped to 0."
        else:
            self.warning = None
