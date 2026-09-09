import copy
from typing import TYPE_CHECKING, List

import numpy as np
from PyQt5.QtCore import QRunnable

from ..graphics import ChunkingRenderer, Frame
from ..video import VideoRenderer
from .gui_utils import global_thread_pool

if TYPE_CHECKING:
    from .app import AldyparenApp


def render_movie_preview_async(app: "AldyparenApp", frame: Frame) -> np.ndarray | str:
    if hasattr(frame, "cached_movie_preview") and frame.cached_movie_preview is not None:
        if isinstance(frame.cached_movie_preview, str) and frame.cached_movie_preview == "wait":
            return "Rendering..."
        else:
            return frame.cached_movie_preview
    else:
        object.__setattr__(frame, "cached_movie_preview", "wait")
        task = MoviePreviewRenderRunnable(app, frame)
        task.setAutoDelete(True)
        global_thread_pool().start(task)
        return "Rendering..."


class MoviePreviewRenderRunnable(QRunnable):
    """Renders frame preview in a separate thread, caches it in Frame object and displays."""

    def __init__(self, app: "AldyparenApp", frame: Frame):
        super().__init__()
        self.app = app
        self.frame = frame

    def run(self):
        image = self.app.movie_frame_renderer.render(self.frame)
        object.__setattr__(self.frame, "cached_movie_preview", image)
        if self.app.frames[self.app.selected_frame_idx] == self.frame:
            self.app.shown_movie_frame_is_invalid = True


class ImageRenderRunnable(QRunnable):

    def __init__(self, app: "AldyparenApp", frame: Frame, renderer: ChunkingRenderer, file_name: str):
        super().__init__()
        self.app = app
        self.frame = frame
        self.renderer = renderer
        self.file_name = file_name

    def run(self):
        self.app.photo_rendering_tasks_count += 1
        try:
            self.renderer.render_picture(self.frame, self.file_name)
        except Exception as e:
            self.app.show_error_message_async(str(e))
        finally:
            self.app.photo_rendering_tasks_count -= 1


def render_video_async(app: "AldyparenApp", width: int, height: int, fps: int, file_name: str):
    renderer = VideoRenderer(width, height, fps, is_aborted=lambda: app.is_exiting)
    thread = VideoRenderRunnable(app, app.frames, renderer, file_name)
    thread.setAutoDelete(True)
    app.active_video_renderer = renderer
    app.video_rendering_tasks_count += 1
    global_thread_pool().start(thread)


class VideoRenderRunnable(QRunnable):

    def __init__(self, app: "AldyparenApp", frames: List[Frame], renderer: VideoRenderer, file_name: str):
        super().__init__()
        self.app = app
        self.frames = copy.copy(frames)
        self.renderer = renderer
        self.file_name = file_name

    def run(self):
        try:
            self.renderer.render_video(self.frames, self.file_name)
        except Exception as e:
            self.app.show_error_message_async(str(e))
        self.app.video_rendering_tasks_count -= 1
