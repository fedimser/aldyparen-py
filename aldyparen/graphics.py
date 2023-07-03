import numpy as np
import numba
import matplotlib
from matplotlib import pyplot as plt
from moviepy.editor import concatenate, ImageClip
from dataclasses import dataclass
from typing import Callable, Union
from PyQt5.QtCore import QThread, QObject
import time


@numba.jit("u1[:,:,:](u4[:,:],u1[:,:])", parallel=True, nogil=True)
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
    def default():
        return ColorPalette.gradient_plus_one('black', 'yellow', 'black', size=101)

    @staticmethod
    def random(size=256):
        return ColorPalette(np.random.randint(0, 256, (size, 3), dtype=np.ubyte))

    @staticmethod
    def grayscale(size=256):
        return ColorPalette.gradient([0, 0, 0], [255, 255, 255], size=size)

    @staticmethod
    def color_to_html(color):
        return '#{:02X}{:02X}{:02X}'.format(color[0], color[1], color[2])


@dataclass(frozen=True)
class Transform:
    center: np.complex128  # Point displayed at the center of screen.
    scale: float  # Width of the frame, in units.
    rotation: float  # radians, about (0, 0), clockwise.

    def translate(self, dx, dy) -> 'Transform':
        return Transform(self.center - dx - dy * 1j, self.scale, self.rotation)

    def rotate_at_point(self, pole, angle) -> 'Transform':
        return Transform(self.center + pole * (np.exp(1j * angle) - 1), self.scale, self.rotation + angle)

    def scale_at_point(self, pole, scale_factor) -> 'Transform':
        return Transform(pole - (pole - self.center) * scale_factor, self.scale * scale_factor, self.rotation)

    def __str__(self):
        return f"{self.center} {self.scale} {self.rotation}"


@dataclass(frozen=True)
class Frame:
    painter: 'Painter'
    transform: Transform
    palette: ColorPalette


class Renderer:
    def __init__(self, width_pxl, height_pxl):
        self.width_pxl = width_pxl
        self.height_pxl = height_pxl

    def render_meshgrid_mono(self, frame: Frame, mgrid_x: np.ndarray, mgrid_y: np.ndarray):
        """Renders monochrome frame for pixels with given coordinates"""
        if hasattr(frame.painter, "render_high_precision"):
            mgrid_shape = mgrid_x.shape
            assert mgrid_y.shape == mgrid_shape
            if len(mgrid_shape) > 1:
                mgrid_x = mgrid_x.reshape((-1,))
                mgrid_y = mgrid_y.reshape((-1,))
            mgrid_x = mgrid_x.astype(np.int16)
            mgrid_y = mgrid_y.astype(np.int16)
            return frame.painter.render_high_precision(self, frame.transform, mgrid_x, mgrid_y).reshape(mgrid_shape)
        else:
            c = frame.transform.center
            upp = frame.transform.scale / self.width_pxl  # units per pixel.
            cs = np.cos(frame.transform.rotation)
            sn = np.sin(frame.transform.rotation)
            x_rot = cs - 1j * sn
            y_rot = sn + 1j * cs
            p_0 = (c.real - upp * 0.5 * (self.width_pxl - 1)) * x_rot + (
                    c.imag + upp * 0.5 * (self.height_pxl - 1)) * y_rot
            k_x = upp * x_rot
            k_y = upp * y_rot
            points = p_0 + mgrid_x * k_x - mgrid_y * k_y
            return frame.painter.paint(points)


class StaticRenderer(Renderer):
    def __init__(self, width_pxl, height_pxl):
        super().__init__(width_pxl, height_pxl)
        self.mgrid_y, self.mgrid_x = np.mgrid[0:height_pxl, 0:width_pxl]

    def render(self, frame):
        pic = self.render_meshgrid_mono(frame, self.mgrid_x, self.mgrid_y)
        return frame.palette.remap(pic)

    def render_picture(self, frame, file_name):
        pic = self.render(frame)
        plt.imsave(file_name, pic)

    def render_video(self, frames, file_name, fps=16):
        clips = []
        for frame in frames:
            clips.append(ImageClip(self.render(frame)).set_duration(1.0 / fps))

        video = concatenate(clips, method="compose")
        video.write_videofile(file_name, fps=fps)


@numba.jit("(u4[:],i2[:],i2[:],u4[:,:])", parallel=True, nogil=True)
def _rearrange_points(points, x, y, output):
    for i in range(len(points)):
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
        print(f"Created renderer with ds={downsample_factor}")

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
            t0 = time.time()
            _rearrange_points(self.mono_pic, self.mgrid_x, self.mgrid_y, pic)
        else:
            small_pic = self.mono_pic[0:self.chunk_size].reshape((self.height_mini, self.width_mini))
            if self.chunks_rendered == 1:
                pic = small_pic
            else:
                pic = small_pic.repeat(self.downsample_factor, axis=0).repeat(self.downsample_factor, axis=1)
                pic = pic[:self.height_pxl, :self.width_pxl]
                assert pic.shape == (self.height_pxl, self.width_pxl)
                l = self.chunks_rendered * self.chunk_size
                _rearrange_points(self.mono_pic[:l], self.mgrid_x[:l], self.mgrid_y[:l], pic)

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


class RenderLoop(QThread):

    def __init__(self, renderer: InteractiveRenderer):
        super().__init__()
        self.renderer = renderer
        self.is_idle = True

    def run(self):
        cs = self.renderer.chunk_size
        while True:
            if self.renderer.frame_to_display is None:
                self.is_idle = True
                time.sleep(0.01)
                continue
            if not (self.renderer.frame_to_display is self.renderer.frame_rendered):
                self.is_idle = False
                self.renderer.mono_pic[:cs] = self.renderer.render_meshgrid_mono(self.renderer.frame_to_display,
                                                                                 self.renderer.mgrid_x[:cs],
                                                                                 self.renderer.mgrid_y[:cs])
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
            self.renderer.mono_pic[cr: cr + cs] = self.renderer.render_meshgrid_mono(self.renderer.frame_rendered,
                                                                                     self.renderer.mgrid_x[cr:cr + cs],
                                                                                     self.renderer.mgrid_y[cr:cr + cs])
            self.renderer.chunks_rendered += 1
        assert self.renderer.chunks_rendered == self.renderer.chunks_count
        self.renderer.need_immediate_update = True
