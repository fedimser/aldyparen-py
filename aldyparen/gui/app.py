import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from typing import List

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, QThreadPool, QCoreApplication
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

from .async_runners import ImageRenderRunnable, render_movie_preview_async
from .main import MainWindow
from .settings import AldyparenSettings
from ..graphics import InteractiveRenderer, StaticRenderer, Transform, Frame, ColorPalette, ChunkingRenderer
from ..mixing import make_animation
from ..painters import MandelbroidPainter, ALL_PAINTERS, PAINTERS_INDEX
from ..video import VideoRenderer, deserialize_movie

APP_NAME = "Aldyparen"
VERSION = "3.0"


class AldyparenApp:
    def __init__(self):
        self.qt_app = QtWidgets.QApplication(sys.argv)
        QCoreApplication.setApplicationName(f"{APP_NAME} {VERSION}")

        self.opened_file_name = None
        self.have_unsaved_changes = False
        self.is_loading_project = False
        self.is_exiting = False
        self.shown_movie_frame_is_invalid = True
        self.error_messages_to_show = []  # type: List[str]

        self.saved_painter_configs = dict()  # TODO: this better store actual painters.
        for painter_class in ALL_PAINTERS:
            painter_name = painter_class.__name__
            self.saved_painter_configs[painter_name] = painter_class().to_object()
        self.selected_painter_class = ALL_PAINTERS[0]
        default_painter = ALL_PAINTERS[0]()
        self.default_transform = Transform.create(scale=4)
        self.work_frame = Frame(default_painter, self.default_transform, ColorPalette.default())

        self.main_window = MainWindow(self)
        self.settings = AldyparenSettings(self)
        self.work_frame_renderer = InteractiveRenderer(self.settings.get_work_view_width(),
                                                       self.settings.get_work_view_height(),
                                                       self.main_window.set_work_frame,
                                                       downsample_factor=self.settings.get_downsample_factor())
        self.movie_frame_renderer = StaticRenderer(self.settings.get_movie_view_width(),
                                                   self.settings.get_movie_view_height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)

        self.frames = []  # type: List[Frame]
        self.selected_frame_idx = -1

        self.photo_rendering_tasks_count = 0
        self.video_rendering_tasks_count = 0
        self.active_video_renderer = None  # type: VideoRenderer | None

    def run(self):
        self.main_window.show()
        self.reset_painter_config()
        self.main_window.update_title()
        self.main_window.on_movie_updated()
        self.main_window.show_palette_preview(self.work_frame.palette)
        self.timer.start(100)
        self.qt_app.exec()

    def on_work_frame_changed(self):
        """Notifies that work frame needs to be re-rendered."""
        self.work_frame_renderer.render_async(self.work_frame)

    def select_painter_type(self, idx):
        painter_class = ALL_PAINTERS[idx]
        if self.is_loading_project:
            return
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

    def reset_painter_config(self):
        self.work_frame = replace(self.work_frame, painter=self.selected_painter_class())
        config = self.work_frame.painter.to_object()
        self.main_window.set_painter_config(json.dumps(config))

    def set_mandelbroid_painter(self, gen_function):
        self.set_painter(MandelbroidPainter(gen_function=gen_function))

    def set_painter(self, painter: 'Painter'):
        self.is_loading_project = True
        self.main_window.combo_painter_type.setCurrentIndex(PAINTERS_INDEX[painter.__class__.__name__])
        self.main_window.set_painter_config(json.dumps(painter.to_object()))
        self.work_frame = replace(self.work_frame, painter=painter)
        self.on_work_frame_changed()
        self.is_loading_project = False

    def reset_work_frame(self):
        self.work_frame_renderer.halt()
        self.work_frame_renderer = InteractiveRenderer(self.settings.get_work_view_width(),
                                                       self.settings.get_work_view_height(),
                                                       self.main_window.set_work_frame,
                                                       downsample_factor=self.settings.get_downsample_factor())
        self.on_work_frame_changed()

    def reset_video_preview(self):
        self.movie_frame_renderer = StaticRenderer(self.settings.get_movie_view_width(),
                                                   self.settings.get_movie_view_height())
        if 0 <= self.selected_frame_idx < len(self.frames):
            frame = self.frames[self.selected_frame_idx]
            object.__setattr__(frame, "cached_movie_preview", None)
            self.shown_movie_frame_is_invalid = True

    def tick(self):
        self.work_frame_renderer.tick()

        # Update UI status labels.
        wf_status_emoji = "✅" if self.work_frame_renderer.renderer_thread.is_idle else "⏳"
        status = wf_status_emoji
        if self.photo_rendering_tasks_count > 0:
            status += f" 📷({self.photo_rendering_tasks_count})"
        if self.video_rendering_tasks_count > 0:
            assert self.active_video_renderer is not None
            status += f" 🎥({self.active_video_renderer.status_string})"
        else:
            self.active_video_renderer = None
        thread_count = QThreadPool.globalInstance().activeThreadCount()
        if thread_count > 0:
            status += f"🧵({thread_count})"
        if hasattr(self.work_frame.painter, "warning") and type(self.work_frame.painter.warning) is str:
            status += "| " + self.work_frame.painter.warning
        self.main_window.show_status(status)

        pos = self.main_window.scene_work_frame.cursor_math_pos
        pos_text = "" if pos is None else "Cursor position: %.15g;%.15g" % (np.real(pos), np.imag(pos))
        self.main_window.label_cursor_position.setText(pos_text)

        if self.shown_movie_frame_is_invalid:
            cur_idx = self.selected_frame_idx
            if 0 <= self.selected_frame_idx < len(self.frames):
                self.main_window.set_movie_frame(render_movie_preview_async(self, self.frames[cur_idx]))
            else:
                self.main_window.scene_movie.clear()
            self.shown_movie_frame_is_invalid = False

        if len(self.error_messages_to_show) > 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(self.error_messages_to_show[0])
            msg.exec()
            self.error_messages_to_show = self.error_messages_to_show[1:]

        if self.main_window.transform_text_is_invalid:
            self.main_window.update_transform_text()

    def update_work_frame_transform(self, new_transform: Transform):
        self.work_frame = replace(self.work_frame, transform=new_transform)
        self.main_window.update_transform_text()
        self.on_work_frame_changed()

    def update_work_frame_palette(self, new_palette: ColorPalette):
        self.work_frame = replace(self.work_frame, palette=new_palette)
        self.on_work_frame_changed()

    def append_movie_frame(self):
        self.frames.append(self.work_frame)
        self.selected_frame_idx = len(self.frames) - 1
        self.have_unsaved_changes = True
        self.main_window.on_movie_updated()

    def replace_movie_frame(self):
        if len(self.frames) == 0:
            return
        self.frames[self.selected_frame_idx] = self.work_frame
        self.have_unsaved_changes = True
        self.main_window.on_movie_updated()

    def remove_last_frames(self, count):
        count = min(count, len(self.frames))
        self.frames = self.frames[:-count]
        self.selected_frame_idx = len(self.frames) - 1
        self.have_unsaved_changes = True
        self.main_window.on_movie_updated()

    def remove_selected_frame(self):
        if len(self.frames) == 0:
            return
        cur_idx = self.selected_frame_idx
        self.frames = self.frames[:cur_idx] + self.frames[cur_idx + 1:]
        if cur_idx >= len(self.frames):
            self.selected_frame_idx = len(self.frames) - 1
        self.have_unsaved_changes = True
        self.main_window.on_movie_updated()

    def clone_selected_frame(self):
        if len(self.frames) == 0:
            return
        self.work_frame = self.frames[self.selected_frame_idx]
        self.have_unsaved_changes = True
        self.on_work_frame_changed()
        self.main_window.ui_handlers_locked = True
        self.main_window.set_painter_config(json.dumps(self.work_frame.painter.to_object()))
        self.main_window.transform_text_is_invalid = True
        self.main_window.ui_handlers_locked = False

    def make_animation(self, length: int):
        assert length >= 2
        cur_idx = self.selected_frame_idx
        assert 0 <= cur_idx < len(self.frames)
        begin_frame = self.frames[cur_idx]
        end_frame = self.work_frame
        if begin_frame.painter.__class__ != end_frame.painter.__class__:
            raise ValueError("Begin and frame use painters of different kinds.")
        anim_frames = make_animation(begin_frame, end_frame, length)
        assert len(anim_frames) == length + 1
        assert anim_frames[0] == begin_frame
        assert anim_frames[-1] == end_frame
        self.frames = self.frames[0:cur_idx] + anim_frames + self.frames[cur_idx + 1:]
        self.selected_frame_idx += length
        self.have_unsaved_changes = True
        self.main_window.on_movie_updated()

    def render_image(self, width, height, file_name=None):
        if file_name is None:
            dir = os.path.join(os.getcwd(), "images")
            if not os.path.exists(dir):
                os.makedirs(dir)
            file_name = datetime.now().isoformat()[:19]
            if type(self.work_frame.painter) is MandelbroidPainter:
                file_name += "[" + self.work_frame.painter.gen_function + "]"
            file_name += ".bmp"
            file_name = os.path.join(dir, file_name)
        renderer = ChunkingRenderer(width, height, is_aborted=lambda: self.is_exiting)
        task = ImageRenderRunnable(self, self.work_frame, renderer, file_name)
        task.setAutoDelete(True)
        QThreadPool.globalInstance().start(task)

    def save_project(self):
        assert self.opened_file_name is not None
        frames_json = []
        prev = None
        for frame in self.frames:
            frames_json.append(frame.serialize(prev=prev))
            prev = frame
        data = {
            "saved_timestamp": datetime.now().isoformat(),
            "version": VERSION,
            "work_frame": self.work_frame.serialize(),
            "frames": frames_json,
            "selected_frame_idx": self.selected_frame_idx,
        }
        with open(self.opened_file_name, "w", encoding="utf-8") as f:
            json.dump(data, f)
        self.have_unsaved_changes = False

    def load_project(self, file_name: str):
        self.is_loading_project = True
        if not os.path.exists(file_name):
            raise ValueError(f"File doesn't exist: {file_name}")
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load work frame to UI.
        self.work_frame = Frame.deserialize(data["work_frame"])
        painter_idx = PAINTERS_INDEX[self.work_frame.painter.__class__.__name__]
        self.main_window.combo_painter_type.setCurrentIndex(painter_idx)
        self.main_window.set_painter_config(json.dumps(self.work_frame.painter.to_object()))
        self.main_window.show_palette_preview(self.work_frame.palette)
        self.main_window.transform_text_is_invalid = True
        self.on_work_frame_changed()

        # Load movie to UI.
        self.frames = deserialize_movie(data["frames"])
        self.selected_frame_idx = data["selected_frame_idx"]
        self.opened_file_name = file_name
        self.have_unsaved_changes = False
        self.main_window.on_movie_updated()
        self.is_loading_project = False

    def new_project(self):
        # Do not reset work frame.
        self.frames = []
        self.selected_frame_idx = -1
        self.have_unsaved_changes = False
        self.main_window.on_movie_updated()
        self.opened_file_name = None

    def get_window_title(self):
        title = f"{APP_NAME} {VERSION}"
        if self.opened_file_name is not None:
            title += " - " + os.path.basename(self.opened_file_name)
        if self.have_unsaved_changes:
            title += "*"
        return title

    def get_selected_frame_info(self):
        cur_frame = self.frames[self.selected_frame_idx]
        return "\n".join([
            "Frame %d of %d" % (self.selected_frame_idx + 1, len(self.frames)),
            cur_frame.painter.__class__.__name__,
            json.dumps(cur_frame.painter.to_object()),
            "Transform: " + str(cur_frame.transform)
        ])

    def show_error_message_async(self, msg):
        self.error_messages_to_show.append(msg)

    def set_palette_color(self, color_idx: int, color: QColor):
        new_colors = np.array(self.work_frame.palette.colors)
        new_colors[color_idx, 0] = color.red()
        new_colors[color_idx, 1] = color.green()
        new_colors[color_idx, 2] = color.blue()
        new_palette = ColorPalette(new_colors)
        self.update_work_frame_palette(new_palette)
        self.main_window.show_palette_preview(new_palette)
