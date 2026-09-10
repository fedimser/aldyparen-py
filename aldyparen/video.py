import os
from time import time
from typing import Callable

from aldyparen.graphics import ChunkingRenderer, Frame
from aldyparen.project import AldyparenProject


class VideoRenderer:
    MAX_MEMORY_USAGE_BYTES = 100_000_000  # 100 MB

    def __init__(
        self, width: int, height: int, fps: int, verbose: bool = False, is_aborted: Callable[[], bool] = lambda: False
    ):
        self.image_renderer = ChunkingRenderer(width, height, chunk_size=100000)
        self.fps = fps
        self.status_string = "Ready"
        self.is_aborted = is_aborted
        self.verbose = verbose

    def render_video(self, frames: list[Frame], file_name: str):
        from moviepy import ImageClip, VideoFileClip, concatenate_videoclips

        if not os.path.splitext(file_name)[1]:
            file_name += ".mp4"
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Split work into parts to limit RAM usage.
        n = len(frames)
        frames_per_part = max(
            0, self.MAX_MEMORY_USAGE_BYTES // (self.image_renderer.width_pxl * self.image_renderer.height_pxl * 3)
        )
        parts_num = (n + frames_per_part - 1) // frames_per_part
        parts: list[tuple[str, list[int]]] = []
        if frames_per_part >= n:
            parts.append((file_name, list(range(n))))
        else:
            for part_id in range(parts_num):
                part_file_name = "{0}_{2:04d}{1}".format(*os.path.splitext(file_name), part_id)
                begin_frame = part_id * frames_per_part
                end_frame = min(begin_frame + frames_per_part, n)
                parts.append((part_file_name, list(range(begin_frame, end_frame))))
        assert [i for _, frame_ids in parts for i in frame_ids] == list(range(n))

        time_start = time()
        self.log("Started")
        frame_ctr = 0
        for part_name, frame_ids in parts:
            clips = []
            for frame_id in frame_ids:
                if self.is_aborted():
                    return
                rendered_frame = self.image_renderer.render(frames[frame_id])
                clips.append(ImageClip(rendered_frame, duration=1.0 / self.fps))
                frame_ctr += 1
                render_rate = (time() - time_start) / frame_ctr
                self.log(f"%d/%d frames, %.1f s/frame" % (frame_ctr, n, render_rate))
            self.log(f"Saving {part_name}...")
            video = concatenate_videoclips(clips, method="compose")
            video.write_videofile(part_name, fps=self.fps, codec="libx264")

        if len(parts) > 1:
            self.log(f"Concatenating parts...")
            clips = [VideoFileClip(part_name) for part_name, _ in parts]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(file_name, codec="libx264")
            self.log(f"Deleting temporary files...")
            for part_name, _ in parts:
                os.remove(part_name)

        self.log("Done")

    def render_movie_from_file(self, input_file: str, output_file: str):
        project = AldyparenProject.load(input_file)
        self.verbose = True
        self.render_video(project.frames, output_file)

    def log(self, text: str):
        self.status_string = text
        if self.verbose:
            print(text)
