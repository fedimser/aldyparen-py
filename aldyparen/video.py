import os
from collections.abc import Callable
from contextlib import ExitStack, closing
from tempfile import TemporaryDirectory
from time import time

from .graphics import ChunkingRenderer, Frame
from .project import AldyparenProject


class VideoRenderer:

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        verbose: bool = False,
        is_aborted: Callable[[], bool] = lambda: False,
        max_memory_bytes: int = 100_000_000,  # 100 MB
    ):
        self.image_renderer = ChunkingRenderer(width, height, chunk_size=100000)
        self.fps = fps
        self.status_string = "Ready"
        self.is_aborted = is_aborted
        self.verbose = verbose
        self.max_memory_bytes = max_memory_bytes

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
            0, self.max_memory_bytes // (self.image_renderer.width_pxl * self.image_renderer.height_pxl * 3)
        )
        if frames_per_part == 0:
            raise ValueError(f"Frame is too large to fit in {self.max_memory_bytes=}")
        parts_num = (n + frames_per_part - 1) // frames_per_part
        with ExitStack() as temporary_files:
            parts: list[tuple[str, list[int]]] = []
            if frames_per_part >= n:
                parts.append((file_name, list(range(n))))
            else:
                temporary_dir = temporary_files.enter_context(TemporaryDirectory(dir=dir_name))
                for part_id in range(parts_num):
                    part_file_name = os.path.join(temporary_dir, f"part_{part_id:04d}.mp4")
                    begin_frame = part_id * frames_per_part
                    end_frame = min(begin_frame + frames_per_part, n)
                    parts.append((part_file_name, list(range(begin_frame, end_frame))))
            assert [i for _, frame_ids in parts for i in frame_ids] == list(range(n))

            time_start = time()
            self.log("Started")
            frame_ctr = 0
            previous_frame: Frame | None = None
            rendered_frame = None
            for part_name, frame_ids in parts:
                with ExitStack() as open_clips:
                    clips = []
                    for frame_id in frame_ids:
                        if self.is_aborted():
                            return
                        frame = frames[frame_id]
                        if frame != previous_frame:
                            rendered_frame = self.image_renderer.render(frame)
                        assert rendered_frame is not None
                        clips.append(
                            open_clips.enter_context(closing(ImageClip(rendered_frame, duration=1.0 / self.fps)))
                        )
                        previous_frame = frame
                        frame_ctr += 1
                        render_rate = (time() - time_start) / frame_ctr
                        self.log(f"{frame_ctr}/{n} frames, {render_rate:.1f} s/frame")
                    self.log(f"Saving {part_name}...")
                    video = open_clips.enter_context(closing(concatenate_videoclips(clips, method="compose")))
                    video.write_videofile(part_name, fps=self.fps, codec="libx264")

            if len(parts) > 1:
                self.log("Concatenating parts...")
                with ExitStack() as open_clips:
                    clips = [open_clips.enter_context(closing(VideoFileClip(part_name))) for part_name, _ in parts]
                    final_clip = open_clips.enter_context(closing(concatenate_videoclips(clips)))
                    final_clip.write_videofile(file_name, codec="libx264")
                self.log("Deleting temporary files...")

        self.log("Done")

    def render_movie_from_file(self, input_file: str, output_file: str):
        project = AldyparenProject.load(input_file)
        self.verbose = True
        self.render_video(project.frames, output_file)

    def log(self, text: str):
        self.status_string = text
        if self.verbose:
            print(text)
