from aldyparen.graphics import Renderer, Transform
from aldyparen.math.hpn import *


@numba.jit("u4(i8[:],i8[:],i4)", nogil=True)
def count_iters(x0, y0, max_iter):
    x2 = np.zeros_like(x0)
    y2 = np.zeros_like(x0)
    w = np.zeros_like(x0)

    for i in range(max_iter):
        x = x2 - y2 + x0
        y = w - x2 - y2 + y0
        x2 = hpn_square(x)
        y2 = hpn_square(y)
        w = hpn_square(x + y)
        s = hpn_normalize(x2 + y2)
        if (s[0] >= 4):
            return i
    return max_iter


# No rotation, so far.

@numba.jit("void(i8[:],i8[:],i2[:],i2[:],i8[:],i4,u4[:])", nogil=True)
def mandelbrot_high_precision_numba(center_x, center_y, mgrid_x, mgrid_y, uphp, max_iter, ans):
    n = len(mgrid_x)
    assert len(mgrid_y) == n
    assert len(ans) == n

    for i in range(n):
        xg = center_x + uphp * (mgrid_x[i])
        yg = center_y - uphp * (mgrid_y[i])
        ans[i] = count_iters(xg, yg, max_iter)


class MandelbrotHighPrecisionPainter:
    """Mandelbrot set with high precision."""

    def __init__(self, max_iter=10, tag="abc"):
        self.max_iter = max_iter
        self.tag = tag

    def to_object(self):
        return {"max_iter": self.max_iter, "tag": self.tag}

    def render_high_precision(self, renderer: Renderer, transform: Transform, mgrid_x: np.ndarray, mgrid_y: np.ndarray,
                              ans: np.ndarray):
        assert mgrid_x.dtype == np.int16
        assert mgrid_y.dtype == np.int16
        assert (mgrid_x.shape == mgrid_y.shape)
        assert len(mgrid_x.shape) == 1
        if not np.allclose(transform.rotation, 0):
            self.warning = "Warning: rotation is ignored!"

        center_x = hpn_from_str(str(np.real(transform.center)))
        center_y = hpn_from_str(str(np.imag(transform.center)))
        # Units per half-pixel.
        uphp = hpn_from_str(str(transform.scale / (2 * renderer.width_pxl)))

        mandelbrot_high_precision_numba(center_x, center_y,
                                        2 * mgrid_x - (renderer.width_pxl - 1),
                                        2 * mgrid_y - (renderer.height_pxl - 1),
                                        uphp, self.max_iter, ans)
