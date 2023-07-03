import copy
import os
import sys
import json
import time

import numpy as np

from .main import MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QThreadPool, QRunnable
from ..graphics import InteractiveRenderer, StaticRenderer, Transform, Frame, ColorPalette
from ..painters import MandelbroidPainter, ALL_PAINTERS
from ..mixing import make_animation
from dataclasses import replace
from datetime import datetime
from typing import List

VERSION = "3.0"


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

        self.photo_rendering_tasks_count = 0
        self.video_rendering_tasks_count = 0

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
        status = wf_status_emoji
        if self.photo_rendering_tasks_count > 0:
            status += f" 📷({self.photo_rendering_tasks_count})"
        if self.video_rendering_tasks_count > 0:
            status += f" 🎥({self.video_rendering_tasks_count})"
        self.main_window.show_status(status)
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

    def append_movie_frame(self):
        self.frames.append(self.work_frame)
        self.selected_frame_idx = len(self.frames) - 1
        self.main_window.on_movie_updated()

    def replace_movie_frame(self):
        if len(self.frames) == 0:
            return
        self.frames[self.selected_frame_idx] = self.work_frame
        self.main_window.on_movie_updated()

    def remove_last_frames(self, count):
        count = min(count, len(self.frames))
        self.frames = self.frames[:-count]
        self.main_window.on_movie_updated()

    def remove_selected_frame(self):
        if len(self.frames) == 0:
            return
        cur_idx = self.selected_frame_idx
        self.frames = self.frames[:cur_idx] + self.frames[cur_idx + 1:]
        if cur_idx >= len(self.frames):
            self.selected_frame_idx = len(self.frames) - 1
        self.main_window.on_movie_updated()

    def clone_selected_frame(self):
        if len(self.frames) == 0:
            return
        self.work_frame = self.frames[self.selected_frame_idx]
        self.on_work_frame_changed()

    def make_animation(self, length: int):
        assert length >= 2
        cur_idx = self.selected_frame_idx
        assert 0 <= cur_idx < len(self.frames)
        begin_frame = self.frames[cur_idx]
        end_frame = self.work_frame
        if begin_frame.painter.__class__ != end_frame.painter.__class__:
            raise ValueError("Begin and frame use painters of different kinds.")
        if begin_frame.palette.colors.shape != end_frame.palette.colors.shape:
            raise ValueError("Begin and frame have palettes of different size.")
        anim_frames = make_animation(begin_frame, end_frame, length)
        assert len(anim_frames) == length + 1
        assert anim_frames[0] == begin_frame
        assert anim_frames[-1] == end_frame
        self.frames = self.frames[0:cur_idx] + anim_frames + self.frames[cur_idx + 1:]
        self.selected_frame_idx += length
        self.main_window.on_movie_updated()

    def render_image(self, width, height):
        dir = os.path.join(os.getcwd(), "images")
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_name = datetime.now().isoformat()[:19]
        if type(self.work_frame.painter) is MandelbroidPainter:
            file_name += "[" + self.work_frame.painter.gen_function + "]"
        file_name += ".bmp"
        file_name = os.path.join(dir, file_name)
        thread = ImageRenderThread(self, self.work_frame, StaticRenderer(width, height), file_name)
        QThreadPool.globalInstance().start(thread)

    def render_video(self, width, height, fps, file_name):
        print(f"Preparing to render video")
        thread = VideoRenderThread(self, self.frames, StaticRenderer(width, height), file_name, fps=fps)
        QThreadPool.globalInstance().start(thread)
        print(f"ImageRenderThread started")

    def serialize_current_project(self) -> str:
        data = {
            "saved_timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "work_frame": self.work_frame.serialize(),
            "frames": [f.serialize() for f in self.frames],
            "selected_frame_idx": self.selected_frame_idx,
        }
        return json.dumps(data)

    def load_project(self, project_json: str):
        data = json.loads(project_json)
        self.work_frame = Frame.deserialize(data["work_frame"])
        self.frames = [Frame.deserialize(f) for f in data["frames"]]
        self.selected_frame_idx = data["selected_frame_idx"]
        self.on_work_frame_changed()
        self.main_window.on_movie_updated()


# Maybe this can be merged with StaticRenderer?
class ImageRenderThread(QRunnable):

    def __init__(self, app: AldyparenApp, frame: Frame, renderer: StaticRenderer, file_name: str, ):
        super().__init__()
        self.app = app
        self.frame = frame
        self.renderer = renderer
        self.file_name = file_name

    def run(self):
        self.app.photo_rendering_tasks_count += 1
        print(f"Start rendering photo")
        time_start = time.time()
        self.renderer.render_picture(self.frame, self.file_name)
        print(f"Rendered f{self.file_name}, time={time.time() - time_start}")
        self.app.photo_rendering_tasks_count -= 1


class VideoRenderThread(QRunnable):

    def __init__(self, app: AldyparenApp, frames: List[Frame], renderer: StaticRenderer, file_name: str, fps: int = 16):
        super().__init__()
        self.app = app
        self.frames = copy.copy(frames)
        self.renderer = renderer
        self.file_name = file_name
        self.fps = fps

    def run(self):
        self.app.video_rendering_tasks_count += 1
        print(f"Start rendering video")
        time_start = time.time()
        self.renderer.render_video(frames=self.frames, file_name=self.file_name, fps=self.fps)
        print(f"Rendered f{self.file_name}, time={time.time() - time_start}")
        self.app.video_rendering_tasks_count -= 1
