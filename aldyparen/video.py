import json
import os
from time import time
from typing import List, Callable

from moviepy.editor import concatenate, ImageClip

from aldyparen.graphics import ChunkingRenderer, Frame


class VideoRenderer:

    def __init__(self, width: int, height: int, fps: int, is_aborted: Callable[[], bool] = lambda: False):
        self.image_renderer = ChunkingRenderer(width, height, chunk_size=100000)
        self.fps = fps
        self.status_string = "Ready"
        self.is_aborted = is_aborted
        self.verbose = False

    def render_video(self, frames: List[Frame], file_name: str):
        time_start = time()
        clips = []
        n = len(frames)
        self.log("Started")
        for i in range(n):
            if self.is_aborted():
                return
            clips.append(ImageClip(self.image_renderer.render(frames[i])).set_duration(1.0 / self.fps))
            render_rate = (time() - time_start) / (i + 1)
            self.log(f"%d/%d frames, %.1f s/frame" % (i + 1, n, render_rate))

        self.log("Saving video...")
        video = concatenate(clips, method="compose")
        dir = os.path.dirname(file_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        video.write_videofile(file_name, fps=self.fps)
        self.log("Done")

    def render_movie_from_file(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = deserialize_movie(data["frames"])
        self.verbose = True
        self.render_video(frames, output_file)

    def log(self, text):
        self.status_string = text
        if self.verbose:
            print(text)


def deserialize_movie(data: List):
    frames = []
    prev = None
    for frame_json in data:
        frame = Frame.deserialize(frame_json, prev=prev)
        frames.append(frame)
        prev = frame
    return frames
