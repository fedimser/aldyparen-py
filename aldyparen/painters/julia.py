import numpy as np
import numba
import warnings

from aldyparen.util import prepare_function


@numba.jit("i4(c16[:],c16[:],i4,f8,u4[:])", nopython=True)
def paint_converged(points_after, attractors, used_colors, tolerance, ans):
    max_colors = len(attractors)  # Equals to JuliaPainter.max_colors+2.
    n = len(points_after)

    for i in range(n):
        z = points_after[i]
        if np.isnan(z) or np.isinf(z) or np.abs(z) > (1 / tolerance):
            ans[i] = 0
            continue
        idx = -1
        for j in range(1, used_colors):
            if np.abs(attractors[j] - z) < tolerance:
                idx = j
                break
        if idx == -1:
            if used_colors < max_colors:
                used_colors += 1
            idx = used_colors - 1
            attractors[idx] = z
        ans[i] = idx
    return used_colors


class JuliaPainter:
    """Paints points of the complex plane based on where they get by applying `func` many times.

    Points z1 and z2 will be painted the same color if `abs(F(z1),F(z2))<tolerance`, where F(z)=f(f(f...(z)..))`, where
    `f` is applied `iters` times.
    If `abs(F(z)) > 1/tolerance`, we assume that z is sent to infinity by this process.

    Colors:
     * All points that go to infinity or NaN will get color 0.
     * Points that converge to some "attractors" will get colors 1, 2 ... max_colors.
     * If there are more attractors than `max_colors`, the rest of them will get the same color `max_colors+1`.

    This can be used to draw [Newton fractal](https://en.wikipedia.org/wiki/Newton_fractal). To get Newton fractal for
    function P, use `JuliaPainter` with `f(z)=z-P(z)/P'(z)`.
    """

    def __init__(self, func="z-(z*z*z-1)/(3*z*z)", iters=100, tolerance=1e-9, max_colors=20):
        """
        :param func: function of variable `z`.
        :param iters: maximal number of iterations for convergence.
        :param tolerance: distance between two points to consider them the same.
        :param max_colors: any number
        """
        self.func = func
        self.iters = iters
        assert 0 < tolerance < 1e-5
        self.tolerance = tolerance
        self.func_prepared = prepare_function(func, variables=['z', 'c'])
        self.max_colors = max_colors
        self.iterate_func = None
        self.warning = None
        self.attractors = np.full((max_colors + 2,), np.inf + 0j, dtype=np.complex128)
        self.used_colors = 1

    def to_object(self):
        return {"func": self.func,
                "iters": self.iters,
                "tolerance": self.tolerance,
                "max_colors": self.max_colors}

    def paint(self, points: np.ndarray, ans: np.ndarray):
        if self.iterate_func is None:
            numba_namespace = {"numba": numba, "np": np}
            stop_tolerance = self.tolerance / 10
            source = "\n".join([
                f'@numba.vectorize("c16(c16)", target="parallel")',
                f'def _iterate(z):',
                f'  for i in range({self.iters}):',
                f'    z2 = {self.func_prepared}',
                f'    if np.abs(z-z2) < {stop_tolerance}: return z2',
                f'    z = z2',
                f'  return z',
            ])
            exec(source, numba_namespace)
            self.iterate_func = numba_namespace["_iterate"]

        warnings.filterwarnings("ignore", message="overflow")
        try:
            points_after = self.iterate_func(points)
        except Exception as e:
            self.warning = f"Error: {e}"
            ans[:] = np.zeros_like(points, dtype=np.uint32)
            return

        self.used_colors = paint_converged(points_after, self.attractors, self.used_colors, self.tolerance, ans)
        if self.used_colors == self.max_colors + 2:
            self.warning = f"Warning! Color limit exceeded, extra colors were mapped to {self.max_colors + 1}."
