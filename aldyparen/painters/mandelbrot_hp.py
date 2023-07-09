from aldyparen.math.hpn import *
import numba
import numpy as np


# This is slower and not used, but left for reference.
@numba.guvectorize([(numba.int64[:], numba.int64[:], numba.uint32[:], numba.uint32[:])], '(n),(n),(m)->(m)',
                   nopython=True)
# @numba.jit("void(i8[:],i8[:],u4[:],u4[:])", nogil=True, nopython=True)
def _count_iters_v1(x0, y0, max_iter, ans):
    x2 = np.zeros_like(x0)
    y2 = np.zeros_like(x0)
    w = np.zeros_like(x0)

    for i in range(max_iter[0]):
        x = x2 - y2 + x0
        y = w - x2 - y2 + y0
        x2 = hpn_square(x)
        y2 = hpn_square(y)
        w = hpn_square(x + y)
        s = x2 + y2
        hpn_normalize_in_place(s)
        if s[0] >= 4:
            ans[0] = i
            return
    ans[0] = max_iter[0]


def _count_iters_vec_v1(x0, y0, max_iter, ans):
    ans = ans.reshape((-1, 1))
    max_iter = np.full_like(ans, max_iter)
    _count_iters_v1(x0, y0, max_iter, ans)


@numba.jit("void(i8[:,:],i8[:,:],u4,u4[:])", nogil=True, nopython=True)
def _count_iters_vec_v2(x0, y0, max_iter, ans):
    hpn_normalize_in_place_vec(x0)
    hpn_normalize_in_place_vec(y0)
    n = len(ans)
    x = np.zeros_like(x0)
    y = np.zeros_like(x0)
    x2 = np.zeros_like(x0)
    y2 = np.zeros_like(x0)
    w = np.zeros_like(x0)
    ans[:] = max_iter

    for i in range(max_iter):
        x[:] = x2 - y2 + x0
        y[:] = w - x2 - y2 + y0
        hpn_mul_vec_inplace(x, x, x2)
        hpn_normalize_in_place_vec(x2)  # x2 = x^2
        hpn_mul_vec_inplace(y, y, y2)
        hpn_normalize_in_place_vec(y2)  # y2 = y^2
        x += y
        hpn_mul_vec_inplace(x, x, w)
        hpn_normalize_in_place_vec(w)  # w = (x+y)^2
        x[:] = x2 + y2
        hpn_normalize_in_place_vec(x)  # x = abs(x+iy)

        done = True
        for j in range(n):
            if ans[j] == max_iter:
                done = False
                if x[j, 0] >= 4:
                    ans[j] = i
        if done:
            break


class MandelbrotHighPrecisionPainter:
    """Mandelbrot set with high precision."""

    def __init__(self, max_iter=10):
        self.max_iter = max_iter

    def to_object(self):
        return {"max_iter": self.max_iter}

    def paint_high_precision(self, points_x: np.ndarray, points_y: np.ndarray, ans: np.ndarray):
        _count_iters_vec_v2(points_x, points_y, self.max_iter, ans)
