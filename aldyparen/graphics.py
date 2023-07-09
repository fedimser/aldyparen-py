import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import matplotlib
import numba
import numpy as np
from PyQt5.QtCore import QThread
from matplotlib import pyplot as plt

from aldyparen.math.hpn import hpn_from_number, hpn_from_str


@numba.jit("u1[:,:,:](u4[:,:],u1[:,:])", parallel=True, nogil=True, nopython=True)
def _numba_remap(pic, colors):
    h, w = pic.shape
    colors_num = colors.shape[0]
    assert colors.shape == (colors_num, 3)
    ans = np.zeros((h, w, 3), dtype=np.ubyte)
    for y in range(h):
        for x in range(w):
            id = pic[y, x]
            for i in range(3):
                ans[y, x, i] = colors[id % colors_num, i]
    return ans


def _to_numpy_color(color) -> np.ndarray:
    """Converts string or RGB list to numpy uint8 array representing RGB"""
    if type(color) is str:
        color = 255 * np.array(matplotlib.colors.to_rgb(color))
    ans = np.array(color, dtype=np.uint8)
    if not ans.shape == (3,):
        raise ValueError("Wrong shape")
    return ans


@dataclass(frozen=True)
class ColorPalette:
    colors: np.ndarray

    def __post_init__(self):
        assert self.colors.shape == (self.colors.shape[0], 3)

    def remap(self, pic):
        return _numba_remap(pic, self.colors)

    @staticmethod
    def gradient(start_color, end_color, size=256):
        colors = np.empty((size, 3), dtype=np.uint8)
        start_color = _to_numpy_color(start_color)
        end_color = _to_numpy_color(end_color)
        for i in range(size):
            a = i / (size - 1)
            colors[i, :] = np.round((1 - a) * start_color + a * end_color)
        return ColorPalette(colors)

    @staticmethod
    def gradient_plus_one(start_color, end_color, extra_color, size=256):
        colors = np.empty((size, 3), dtype=np.uint8)
        colors[:size - 1, :] = ColorPalette.gradient(start_color, end_color, size=size - 1).colors
        colors[-1, :] = _to_numpy_color(extra_color)
        return ColorPalette(colors)

    @staticmethod
    def categorical(colors_html):
        size = len(colors_html)
        colors = np.empty((size, 3), dtype=np.uint8)
        for i in range(size):
            colors[i, :] = _to_numpy_color(colors_html[i])
        return ColorPalette(colors)

    @staticmethod
    def default():
        return ColorPalette.categorical(
            ['white', 'yellow', 'purple', 'orange', 'lightblue', 'red', 'gray', 'green', 'black'])

    @staticmethod
    def random(size=256):
        return ColorPalette(np.random.randint(0, 256, (size, 3), dtype=np.uint8))

    @staticmethod
    def grayscale(size=256):
        return ColorPalette.gradient([0, 0, 0], [255, 255, 255], size=size)

    @staticmethod
    def color_to_html(color):
        return '#{:02X}{:02X}{:02X}'.format(color[0], color[1], color[2])

    def serialize(self) -> str:
        return self.colors.tobytes().hex()

    @staticmethod
    def deserialize(data: str) -> 'ColorPalette':
        colors = np.frombuffer(bytes.fromhex(data), dtype=np.uint8).reshape((-1, 3))
        return ColorPalette(np.array(colors))

    def __eq__(self, other: 'ColorPalette'):
        return np.array_equal(self.colors, other.colors)


LN_10 = np.log(10)


@dataclass(frozen=True)
class Transform:
    center: np.complex128  # Point displayed at the center of the frame.
    scale_log10: float  # Base-10 logarithm of the frame width (in math units).
    rotation: float  # Radians, about frame center, counterclockwise.

    @staticmethod
    def create(*, center=None, center_x: float | str = None, center_y: float | str = None,
               scale_log10=None,  scale=None,
               rotation=None, rotation_deg=None) -> 'Transform':
        if scale is not None:
            scale_log10 = np.log10(scale)
        if rotation_deg is not None:
            rotation = (rotation_deg / 180) * np.pi
        if center_x is not None and center_y is not None:
            center = np.complex128(float(center_x) + 1j * float(center_y))

        transform = Transform(np.complex128(center or 0.0), scale_log10 or 0.0, rotation or 0.0)
        if type(center_x) is str:
            object.__setattr__(transform, "center_x_str", center_x)
        if type(center_y) is str:
            object.__setattr__(transform, "center_y_str", center_y)
        return transform

    def translate(self, delta) -> 'Transform':
        return Transform(self.center - delta * self._k(), self.scale_log10, self.rotation)

    def rotate_and_scale_at(self, rel_screen_point, scale_factor=1.0, angle=0.0) -> 'Transform':
        old_k = self._k()
        new_scale_log_10 = self.scale_log10 + np.log10(scale_factor)
        new_rotation = self.rotation + angle
        new_k = np.exp(LN_10 * new_scale_log_10 - 1j * new_rotation)
        new_center = self.center + (old_k - new_k) * rel_screen_point
        return Transform(new_center, new_scale_log_10, new_rotation)

    def _k(self):
        return np.exp(LN_10 * self.scale_log10 - 1j * self.rotation)

    def map_screen_to_math(self, screen_point: np.complex128) -> np.complex128:
        return self.center + screen_point * self._k()

    def __str__(self):
        rot_deg = (self.rotation / np.pi * 180) % 360
        scale_exp = int(np.floor(self.scale_log10))
        scale_base = np.power(10, self.scale_log10 - scale_exp)
        scale_str = "%.2fe%d" % (scale_base, scale_exp)
        return "c=(%.5e %.5e) s=%s r=%.1f°" % (
            self.center.real, self.center.imag, scale_str, rot_deg)

    def serialize(self) -> List[float]:
        return [self.center.real, self.center.imag, self.scale_log10, self.rotation]

    @staticmethod
    def deserialize(data: List[float]) -> 'Transform':
        assert len(data) == 4
        return Transform(np.complex128(data[0] + 1j * data[1]), data[2], data[3])

    def __eq__(self, other: 'Transform'):
        return np.isclose(self.center, other.center) and np.isclose(self.scale_log10, other.scale_log10) and np.isclose(
            self.rotation, other.rotation)

    def rotation_deg(self):
        return 180 * self.rotation / np.pi


@dataclass(frozen=True)
class Frame:
    painter: 'Painter'
    transform: Transform
    palette: ColorPalette

    def serialize(self, prev: 'Frame' = None):
        data = {
            "tr": self.transform.serialize(),
        }
        if prev is not None and prev.painter == self.painter:
            data["pn"] = "prev"
        else:
            data["pn"] = self.painter.__class__.__name__
            data["pt"] = self.painter.to_object()
        if prev is not None and prev.palette == self.palette:
            data["pl"] = "prev"
        else:
            data["pl"] = self.palette.serialize()
        return data

    @staticmethod
    def deserialize(data: Dict, prev: 'Frame' = None) -> 'Frame':
        from aldyparen.painters import Painter
        if data["pn"] == "prev":
            painter = prev.painter
        else:
            painter = Painter.deserialize(data["pn"], data["pt"])
        if data["pl"] == "prev":
            palette = prev.palette
        else:
            palette = ColorPalette.deserialize(data["pl"])
        return Frame(
            painter=painter,
            transform=Transform.deserialize(data["tr"]),
            palette=palette
        )


class Renderer:
    def __init__(self, width_pxl, height_pxl):
        self.width_pxl = width_pxl
        self.height_pxl = height_pxl

    def render_meshgrid_mono(self, frame: Frame, mgrid_x: np.ndarray, mgrid_y: np.ndarray, ans: np.ndarray):
        """Renders monochrome frame for pixels with given coordinates"""
        frame.painter.warning = None
        assert ans.shape == mgrid_x.shape == mgrid_y.shape
        if len(mgrid_x.shape) > 1:
            mgrid_x = mgrid_x.reshape((-1,))
            mgrid_y = mgrid_y.reshape((-1,))
            ans = ans.reshape((-1,))
        if mgrid_x.dtype != np.int16:
            mgrid_x = mgrid_x.astype(np.int16)
        if mgrid_y.dtype != np.int16:
            mgrid_y = mgrid_y.astype(np.int16)
        tr = frame.transform
        rot_cos = np.cos(tr.rotation)
        rot_sin = np.sin(tr.rotation)
        w = self.width_pxl
        h = self.height_pxl

        if hasattr(frame.painter, "paint_high_precision"):
            center_x_str = tr.center_x_str if hasattr(tr, "center_x_str") else str(tr.center.real)
            center_y_str = tr.center_y_str if hasattr(tr, "center_y_str") else str(tr.center.imag)
            scale_exp = int(np.floor(tr.scale_log10))
            scale_base = np.power(10, tr.scale_log10 - scale_exp)
            prec = max(len(center_x_str), len(center_y_str), -scale_base) // 8 + 5
            center_x = hpn_from_str(center_x_str, prec=prec)
            center_y = hpn_from_str(center_y_str, prec=prec)

            uphp = scale_base / (2 * self.width_pxl)  # Units per half-pixel.
            k_cos = hpn_from_str(str(uphp * rot_cos), prec=prec, extra_power_10=scale_exp)
            k_sin = hpn_from_str(str(uphp * rot_sin), prec=prec, extra_power_10=scale_exp)
            mgrid_x = 2 * mgrid_x - (w - 1)
            mgrid_y = 2 * mgrid_y - (h - 1)
            points_x = center_x.reshape((1, prec)) + np.outer(mgrid_x, k_cos) - np.outer(mgrid_y, k_sin)
            points_y = center_y.reshape((1, prec)) - np.outer(mgrid_x, k_sin) - np.outer(mgrid_y, k_cos)
            frame.painter.paint_high_precision(points_x, points_y, ans)
        else:
            mgrid = ((mgrid_x - 0.5 * (w - 1)) - 1j * (mgrid_y - 0.5 * (h - 1))) / self.width_pxl
            points = tr.map_screen_to_math(mgrid)
            frame.painter.paint(points, ans)


class StaticRenderer(Renderer):
    def __init__(self, width_pxl, height_pxl):
        super().__init__(width_pxl, height_pxl)
        self.mgrid_y, self.mgrid_x = np.mgrid[0:height_pxl, 0:width_pxl]

    def render(self, frame):
        pic = np.empty((self.height_pxl, self.width_pxl), dtype=np.uint32)
        self.render_meshgrid_mono(frame, self.mgrid_x, self.mgrid_y, pic)
        return frame.palette.remap(pic)


class ChunkingRenderer(Renderer):
    """Renders picture in chunks of `chunk_size` pixels. Can be aborted between chunks."""

    def __init__(self, width_pxl, height_pxl, chunk_size=100000, is_aborted: Callable[[], bool] = lambda: False):
        super().__init__(width_pxl, height_pxl)
        self.chunk_size = chunk_size
        self.chunks_count = int(np.ceil((width_pxl * height_pxl) / self.chunk_size))
        mgrid_y, mgrid_x = np.mgrid[0:height_pxl, 0:width_pxl]
        self.mgrid_x = mgrid_x.reshape((-1,))
        self.mgrid_y = mgrid_y.reshape((-1,))
        self.is_aborted = is_aborted

    def render(self, frame: Frame) -> np.ndarray:
        pic = np.empty((self.width_pxl * self.height_pxl), dtype=np.uint32)
        cs = self.chunk_size
        for i in range(self.chunks_count):
            if self.is_aborted():
                break
            st = i * cs
            self.render_meshgrid_mono(frame, self.mgrid_x[st: st + cs], self.mgrid_y[st:st + cs], pic[st: st + cs])
        return frame.palette.remap(pic.reshape(self.height_pxl, self.width_pxl))

    def render_picture(self, frame, file_name):
        pic = self.render(frame)
        plt.imsave(file_name, pic)


@numba.jit("(u4[:],i2[:],i2[:],u4[:,:])", parallel=True, nogil=True)
def _rearrange_points(points, x, y, output):
    for i in numba.prange(len(points)):
        output[y[i]][x[i]] = points[i]


class InteractiveRenderer(Renderer):
    def __init__(self, width_pxl, height_pxl, ui_callback: Callable[[np.ndarray], None], downsample_factor=2):
        super().__init__(width_pxl, height_pxl)
        self.ui_callback = ui_callback
        self.downsample_factor = downsample_factor
        self.frame_rendered = None
        self.frame_to_display = None
        self.frame_displayed = None
        self.chunks_rendered = 0
        self.chunks_displayed = 0
        self.mono_pic = np.empty((width_pxl * height_pxl,), dtype=np.uint32)
        self.last_ui_update_time = 0
        self.need_immediate_update = False

        # Prepare meshgrid such that first chunk is "mini" image.
        mgx = [[], []]
        mgy = [[], []]
        p = downsample_factor // 2
        for y in range(height_pxl):
            for x in range(width_pxl):
                i = 1
                if y % downsample_factor == p and x % downsample_factor == p:
                    i = 0
                mgx[i].append(x)
                mgy[i].append(y)
        self.mgrid_x = np.array(sum(mgx, []), dtype=np.int16)
        self.mgrid_y = np.array(sum(mgy, []), dtype=np.int16)
        self.width_mini = int(np.ceil(width_pxl / downsample_factor))
        self.height_mini = int(np.ceil(height_pxl / downsample_factor))
        self.chunk_size = self.width_mini * self.height_mini
        self.chunks_count = int(np.ceil((width_pxl * height_pxl) / self.chunk_size))
        self.empty_pic = np.zeros((self.height_mini, self.width_mini, 3), dtype=np.ubyte)  # Black rectangle.
        self.renderer_thread = RenderLoop(self)
        self.renderer_thread.start()

    def tick(self):
        """Shows rendered picture in UI, if it changed since last UI update."""
        if not self.need_immediate_update and time.time() - self.last_ui_update_time < 0.5:
            return
        if self.frame_displayed is self.frame_rendered and self.chunks_rendered == self.chunks_displayed:
            return
        self.show_rendered()

    def show_rendered(self):
        """Builds colored 2d picture for currently rendered mono_pic."""
        if self.chunks_rendered == 0:
            return self.empty_pic
        elif self.chunks_rendered == self.chunks_count:
            pic = np.empty((self.height_pxl, self.width_pxl), dtype=np.uint32)
            _rearrange_points(self.mono_pic, self.mgrid_x, self.mgrid_y, pic)
        else:
            small_pic = self.mono_pic[0:self.chunk_size].reshape((self.height_mini, self.width_mini))
            if self.chunks_rendered == 1:
                pic = small_pic
            else:
                pic = small_pic.repeat(self.downsample_factor, axis=0).repeat(self.downsample_factor, axis=1)
                pic = pic[:self.height_pxl, :self.width_pxl]
                assert pic.shape == (self.height_pxl, self.width_pxl)
                length = self.chunks_rendered * self.chunk_size
                _rearrange_points(self.mono_pic[:length], self.mgrid_x[:length], self.mgrid_y[:length], pic)

        pic = self.frame_rendered.palette.remap(pic)
        self.ui_callback(pic)
        self.chunks_displayed = self.chunks_rendered
        self.frame_displayed = self.frame_rendered
        self.last_ui_update_time = time.time()
        self.need_immediate_update = False

    def render_async(self, frame: Frame):
        self.frame_to_display = frame
        # If only palette changed, skip re-rendering.
        fr = self.frame_rendered
        if fr is not None and fr.painter is frame.painter and fr.transform is frame.transform:
            self.frame_rendered = frame
        self.tick()

    def halt(self):
        self.renderer_thread.requestInterruption()


class RenderLoop(QThread):

    def __init__(self, renderer: InteractiveRenderer):
        super().__init__()
        self.renderer = renderer
        self.is_idle = True

    def run(self):
        cs = self.renderer.chunk_size
        while True:
            if self.isInterruptionRequested():
                return
            if self.renderer.frame_to_display is None:
                self.is_idle = True
                time.sleep(0.01)
                continue
            if not (self.renderer.frame_to_display is self.renderer.frame_rendered):
                self.is_idle = False
                self.renderer.render_meshgrid_mono(self.renderer.frame_to_display,
                                                   self.renderer.mgrid_x[:cs],
                                                   self.renderer.mgrid_y[:cs],
                                                   self.renderer.mono_pic[:cs])
                self.renderer.frame_rendered = self.renderer.frame_to_display
                self.renderer.chunks_rendered = 1
                self.renderer.need_immediate_update = True
                continue
            assert self.renderer.chunks_rendered >= 1
            if self.renderer.chunks_rendered == self.renderer.chunks_count:
                self.is_idle = True
                time.sleep(0.001)
                continue

            cr = self.renderer.chunks_rendered * self.renderer.chunk_size
            self.renderer.render_meshgrid_mono(self.renderer.frame_rendered,
                                               self.renderer.mgrid_x[cr:cr + cs],
                                               self.renderer.mgrid_y[cr:cr + cs],
                                               self.renderer.mono_pic[cr: cr + cs])
            self.renderer.chunks_rendered += 1
        assert self.renderer.chunks_rendered == self.renderer.chunks_count
        self.renderer.need_immediate_update = True
