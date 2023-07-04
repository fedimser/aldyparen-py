import numpy as np
from PyQt5.QtCore import QRunnable, QThreadPool

from ..graphics import Frame


def render_movie_preview_async(app: 'AldyparenApp', frame: Frame) -> np.ndarray | str:
    if hasattr(frame, "cached_movie_preview"):
        if frame.cached_movie_preview == "wait":
            return "Rendering..."
        else:
            return frame.cached_movie_preview
    else:
        object.__setattr__(frame, "cached_movie_preview", "wait")
        thread = MoviePreviewRenderThread(app, frame)
        QThreadPool.globalInstance().start(thread)
        return "Rendering..."


class MoviePreviewRenderThread(QRunnable):
    """Renders frame preview in a separate thread, caches it in Frame object and displays."""

    def __init__(self, app: 'AldyparenApp', frame: Frame):
        super().__init__()
        self.app = app
        self.frame = frame

    def run(self):
        print(f"Start rendering movie frame")
        image = self.app.movie_frame_renderer.render(self.frame)
        object.__setattr__(self.frame, "cached_movie_preview", image)
        if self.app.frames[self.app.selected_frame_idx] == self.frame:
            self.app.need_update_movie_in_tick = True
        print(f"Done rendering movie frame")
