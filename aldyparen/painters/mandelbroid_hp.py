import numba
import numpy as np
from numpy.typing import NDArray

from aldyparen.math.complex_hpn import is_on_or_outside_circle
from aldyparen.math.hpn import Hpn
from aldyparen.math.hpn_compiler import compile_expression_hpcn


class MandelbroidHighPrecisionPainter:
    """Renders mandelbrot-like fractal with high precision."""

    def __init__(
        self,
        gen_function: str = "z*z+c",
        max_iter: int = 100,
        radius: float = 2,
        precision: int = 4,
    ):
        assert 1 <= max_iter <= 1000000, "bad max_iter"
        self.gen_function = gen_function
        self.max_iter = max_iter
        self.radius = radius
        self.precision = precision

        gen_func = compile_expression_hpcn(gen_function, var_names=["z", "c"], precision=precision)
        radius_squared = Hpn.from_number(radius**2, prec=precision).digits

        # u4(i8[:],i8[:])
        @numba.njit(nogil=True, inline="always")
        def _paint_func(c_re: NDArray[np.int64], c_im: NDArray[np.int64]) -> np.uint32:
            z_re = np.zeros_like(c_re)
            z_im = np.zeros_like(c_im)
            # assert len(radius_squared) == len(z_re)
            for i in range(max_iter):
                # print("Iter ", i, "before z=", ComplexHpn.from_raw((z_re, z_im)).to_complex())
                z_re, z_im = gen_func((z_re, z_im), (c_re, c_im))
                # print("Iter ", i, "after z=", ComplexHpn.from_raw((z_re, z_im)).to_complex())
                if is_on_or_outside_circle((z_re, z_im), radius_squared):
                    return i
            return max_iter

        # void(i8[:,:],i8[:,:],u4[:]
        @numba.njit(nogil=True)
        def _paint_func_2(
            points_x: NDArray[np.int64],
            points_y: NDArray[np.int64],
            ans: NDArray[np.uint32],
        ):
            n = len(ans)
            for i in numba.prange(n):
                ans[i] = _paint_func(points_x[i, :], points_y[i, :])

        self.paint_func = _paint_func_2

    def to_object(self):
        return {
            "gen_function": self.gen_function,
            "radius": self.radius,
            "max_iter": self.max_iter,
            "precision": self.precision,
        }

    def paint_high_precision(self, points_x: np.ndarray, points_y: np.ndarray, ans: np.ndarray):
        self.paint_func(points_x, points_y, ans)
