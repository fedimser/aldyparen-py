import copy
import sys
import json
import time

import numpy as np

from .main import MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QTimer, QThread
from ..graphics import InteractiveRenderer, StaticRenderer, Transform, Frame, ColorPalette
from ..painters import MandelbroidPainter, AxisPainter, ALL_PAINTERS
from dataclasses import replace
from datetime import datetime
from typing import List


class AldyparenApp:
    def __init__(self):
        # self.preview_renderer = StaticRenderer(480, 270)

        self.qt_app = QtWidgets.QApplication(sys.argv)

        self.saved_painter_configs = dict()  # TODO: this better store actual painters.
        for painter_class in ALL_PAINTERS:
            painter_name = painter_class.__name__
            self.saved_painter_configs[painter_name] = painter_class().to_object()
        self.selected_painter_class = ALL_PAINTERS[0]
        default_painter = ALL_PAINTERS[0]()
        self.default_transform = Transform(0, 4, 0.0)
        self.work_frame = Frame(default_painter, self.default_transform, ColorPalette.default())

        self.main_window = MainWindow(self)
        self.work_frame_renderer = InteractiveRenderer(480, 270, self.main_window.set_work_frame)
        self.movie_frame_renderer = StaticRenderer(240, 135)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)

        self.frames = []  # type: List[Frame]
        self.selected_frame_idx = -1

        # TODO: use QThreadPool isntead.
        self.very_stupid_thread_pool = []

    def run(self):
        self.main_window.show()
        self.reset_config()
        self.timer.start(25)
        self.qt_app.exec()

    def on_work_frame_changed(self):
        """Notifies that work frame needs to be re-rendered."""
        self.work_frame_renderer.render_async(self.work_frame)

    def select_painter_type(self, idx):
        painter_class = ALL_PAINTERS[idx]
        self.selected_painter_class = painter_class
        config = self.saved_painter_configs[painter_class.__name__]
        self.main_window.set_painter_config(json.dumps(config))
        self.work_frame = replace(self.work_frame, painter=painter_class(**config))
        self.on_work_frame_changed()

    def set_painter_config(self, config_json):
        config = {}
        try:
            config = json.loads(config_json)
        except Exception as e:
            self.main_window.set_label_painter_status("Invalid JSON")
            return

        try:
            new_painter = self.selected_painter_class(**config)
        except Exception as e:
            self.main_window.set_label_painter_status(str(e))
            return

        self.work_frame = replace(self.work_frame, painter=new_painter)
        self.saved_painter_configs[self.selected_painter_class.__name__] = config
        self.main_window.set_label_painter_status("OK")
        self.on_work_frame_changed()

    def reset_transform(self):
        self.update_work_frame_transform(self.default_transform)

    def reset_config(self):
        self.work_frame = replace(self.work_frame, painter=self.selected_painter_class())
        config = self.work_frame.painter.to_object()
        self.main_window.set_painter_config(json.dumps(config))

    # TODO: default downsample factor should be in settings.
    def reset_work_frame(self, downsample_factor=3):
        self.work_frame_renderer.renderer_thread.quit()
        self.work_frame_renderer = InteractiveRenderer(480, 270, self.main_window.set_work_frame,
                                                       downsample_factor=downsample_factor)
        self.on_work_frame_changed()

    def tick(self):
        self.work_frame_renderer.tick()

        # Update UI status labels.
        wf_status_emoji = "✅" if self.work_frame_renderer.renderer_thread.is_idle else "⏳"
        self.main_window.show_status(wf_status_emoji)
        pos = self.main_window.scene_work_frame.cursor_math_pos
        pos_text = "" if pos is None else "Cursor position: %.4g;%.4g" % (np.real(pos), np.imag(pos))
        self.main_window.label_cursor_position.setText(pos_text)
        self.main_window.label_transform_info.setText(str(self.work_frame.transform))

    def update_work_frame_transform(self, new_transform: Transform):
        self.work_frame = replace(self.work_frame, transform=new_transform)
        self.on_work_frame_changed()

    def update_work_frame_palette(self, new_palette: ColorPalette):
        self.work_frame = replace(self.work_frame, palette=new_palette)
        self.on_work_frame_changed()

    def export_image(self):
        print(f"Preparing to render photo")
        file_name = "images/" + datetime.now().isoformat()[:19]
        if type(self.work_frame.painter) is MandelbroidPainter:
            file_name += "[" + self.work_frame.painter.gen_function + "]"
        file_name += ".bmp"
        print(f"file name:", file_name)
        thread = ImageRenderThread(self.work_frame, StaticRenderer(3840, 2160), file_name)
        thread.start()
        self.very_stupid_thread_pool.append(thread)
        print(f"ImageRenderThread started")

    def make_animation(self, length):
        # return "Not implemented"
        self.main_window.on_movie_updated()

    def append_movie_frame(self):
        self.frames.append(self.work_frame)
        self.selected_frame_idx = len(self.frames) - 1
        self.main_window.on_movie_updated()


# Maybe this can be merged with StaticRenderer?
class ImageRenderThread(QThread):

    def __init__(self, frame: Frame, renderer: StaticRenderer, file_name: str):
        super().__init__()
        self.frame = frame
        self.renderer = renderer
        self.file_name = file_name

    def run(self):
        print(f"Start rendering photo")
        time_start = time.time()
        self.renderer.render_picture(self.frame, self.file_name)
        print(f"Rendered f{self.file_name}, time={time.time() - time_start}")
