# Mandelbrot set with high precision.

from aldyparen import Renderer, Transform
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
def mandelbrot_high_precision_numba(center_x, center_y, mgrid_x, mgrid_y, upp, max_iter, ans):
    n = len(mgrid_x)
    assert len(mgrid_y) == n
    assert len(ans) == n

    for i in range(n):
        xg = center_x + upp * (mgrid_x[i])
        yg = center_y - upp * (mgrid_y[i])
        ans[i] = count_iters(xg, yg, max_iter)


class MadnelbrotHighPrecisionPainter:
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
        self.warning = None
        if not np.allclose(transform.rotation, 0):
            self.warning = "Warning: rotation is ignored!"
        # TODO: make this more universal.
        # TODO: account for the 1/2 error - can supply doubled mgrid, and different upp.

        center_x = hpn_from_str(str(np.real(transform.center)))
        center_y = hpn_from_str(str(np.imag(transform.center)))
        # Units per pixel.
        upp = hpn_from_str(str(transform.scale / renderer.width_pxl))

        w = renderer.width_pxl
        h = renderer.height_pxl
        mandelbrot_high_precision_numba(center_x, center_y, mgrid_x - w // 2, mgrid_y - h // 2, upp,
                                        self.max_iter, ans)
