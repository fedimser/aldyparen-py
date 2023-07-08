from aldyparen.math.hpn import *


@numba.jit("void(i8[:,:],i8[:,:],u4,u4[:])", nogil=True, nopython=True)
def _count_iters_vec(x0, y0, max_iter, ans):
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
        _count_iters_vec(points_x, points_y, self.max_iter, ans)
