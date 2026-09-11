import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
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
        max_memory_bytes: int = 100_000_00,  # 100 MB
        skip_existing_parts: bool = False,
    ):
        assert 0 < width < 10000
        assert 0 < height < 10000
        assert 0 < fps < 100
        self.image_renderer = ChunkingRenderer(width, height, chunk_size=100000)
        self.fps = fps
        self.status_string = "Ready"
        self.is_aborted = is_aborted
        self.verbose = verbose
        self.max_memory_bytes = max_memory_bytes
        self.skip_existing_parts = skip_existing_parts

        # State for logging progress.
        self.total_frames = 0
        self.rendered_frame_counter = 0
        self.time_redering_started = 0

    def _render_clip(self, frames: list[Frame], file_name: str) -> bool:
        # Renders sequence of frames into given file.
        # Internal helper: assumes that directory exists, all frames fit in memory.
        from moviepy import ImageClip, concatenate_videoclips

        if self.skip_existing_parts and os.path.exists(file_name):
            self.log(f"Not rendering existing {file_name}")
            self.total_frames -= len(frames)
            return True

        assert os.path.splitext(file_name)[1] == ".mp4"
        assert len(frames) > 0

        n = len(frames)
        clips = []
        video = None
        clip_length = 0  # Length of run of identical frames.
        try:
            for i, frame in enumerate(frames):
                if self.is_aborted():
                    return False
                clip_length += 1
                if i != n - 1 and frame == frames[i + 1]:
                    continue

                rendered_frame = self.image_renderer.render(frame)
                clips.append(ImageClip(rendered_frame, duration=clip_length / self.fps))
                self.log_progress(clip_length)
                clip_length = 0
            self.log(f"Saving {file_name}...")
            video = concatenate_videoclips(clips, method="compose")
            video.write_videofile(file_name, fps=self.fps, codec="libx264")
            return True
        finally:
            if video is not None:
                video.close()
            for clip in clips:
                clip.close()

    def render_video(self, frames: list[Frame], file_name: str):
        from moviepy.config import FFMPEG_BINARY

        if not os.path.splitext(file_name)[1] == ".mp4":
            raise ValueError("file extension must be .mp4")
        dir_name = os.path.dirname(file_name) or "."
        os.makedirs(dir_name, exist_ok=True)

        self.total_frames = len(frames)
        self.rendered_frame_counter = 0
        self.time_redering_started = time()

        # Split work into parts to limit RAM usage.
        n = len(frames)
        if n == 0:
            raise ValueError("Cannot render a video with no frames")
        frames_per_part = max(
            0, self.max_memory_bytes // (self.image_renderer.width_pxl * self.image_renderer.height_pxl * 3)
        )
        if frames_per_part == 0:
            raise ValueError(f"Frame is too large to fit in {self.max_memory_bytes=}")
        parts_num = (n + frames_per_part - 1) // frames_per_part

        self.log("Started")
        if parts_num == 1:
            # Video can be rendered in a single clip.
            if self._render_clip(frames, file_name):
                self.log("Done")
            return

        # Render video in parts.
        parts_dir = os.path.splitext(file_name)[0] + "_parts"
        if os.path.exists(parts_dir):
            if not self.skip_existing_parts:
                if os.path.isdir(parts_dir):
                    shutil.rmtree(parts_dir)
                else:
                    os.remove(parts_dir)
                os.makedirs(parts_dir)
        else:
            os.makedirs(parts_dir)
        part_file_names = []
        for part_id in range(parts_num):
            begin_frame = part_id * frames_per_part
            end_frame = min(begin_frame + frames_per_part, n)
            part_file_name = os.path.join(parts_dir, f"part_{part_id:04d}.mp4")
            if not self._render_clip(frames[begin_frame:end_frame], part_file_name):
                return
            part_file_names.append(part_file_name)

        concat_file_name = os.path.abspath(os.path.join(parts_dir, "concat.txt"))
        with open(concat_file_name, "w", encoding="utf-8") as concat_file:
            for part_file_name in part_file_names:
                concat_file.write(f"file '{os.path.basename(part_file_name)}'\n")

        command = [
            FFMPEG_BINARY,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file_name,
            "-c",
            "copy",
            os.path.abspath(file_name),
        ]
        printable_command = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        self.log(f"Concatenating parts with command: {printable_command}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            self.log("FFmpeg concatenation failed.")
            return
        assert os.path.exists(file_name)

        print("Cleaning up parts...")
        shutil.rmtree(parts_dir)

        self.log("Done")

    def render_movie_from_file(self, input_file: str, output_file: str):
        project = AldyparenProject.load(input_file)
        self.verbose = True
        self.render_video(project.frames, output_file)

    def log_progress(self, new_frames_rendered: int):
        self.rendered_frame_counter += new_frames_rendered
        render_rate = (time() - self.time_redering_started) / self.rendered_frame_counter
        self.log(f"{self.rendered_frame_counter}/{self.total_frames} frames, {render_rate:.1f} s/frame")

    def log(self, text: str):
        self.status_string = text
        if self.verbose:
            print(text)
