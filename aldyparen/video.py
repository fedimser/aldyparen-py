from time import time
from typing import List, Callable

from moviepy.editor import concatenate, ImageClip

from aldyparen.graphics import ChunkingRenderer, Frame, StaticRenderer


class VideoRenderer:

    def __init__(self, width: int, height: int, fps: int, is_aborted: Callable[[], bool] = lambda: False):
        self.image_renderer = ChunkingRenderer(width, height, chunk_size=100000)
        self.fps = fps
        self.status_string = "Ready"
        self.is_aborted = is_aborted

    def render_video(self, frames: List[Frame], file_name: str):
        time_start = time()
        clips = []
        n = len(frames)
        self.status_string = "Started"
        for i in range(n):
            if self.is_aborted():
                return
            clips.append(ImageClip(self.image_renderer.render(frames[i])).set_duration(1.0 / self.fps))
            render_rate = (time() - time_start) / (i + 1)
            self.status_string = f"%d/%d frames, %.1f s/frame" % (i + 1, n, render_rate)
        self.status_string = "Saving video..."
        video = concatenate(clips, method="compose")
        video.write_videofile(file_name, fps=self.fps)
        self.status_string = "Done"
