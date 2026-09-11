from pathlib import Path
from unittest.mock import patch

import pytest
from moviepy import VideoFileClip

from aldyparen.graphics import ColorPalette, Frame, Transform
from aldyparen.painters import SierpinskiCarpetPainter
from aldyparen.video import VideoRenderer


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning:moviepy.video.io.ffmpeg_reader"
)
def test_render_video_with_current_moviepy(tmp_path: Path):
    renderer = VideoRenderer(8, 8, fps=1)
    palette = ColorPalette.categorical(["black", "white"])
    frames = [
        Frame(SierpinskiCarpetPainter(depth=1), Transform.create(scale=1), palette),
        Frame(SierpinskiCarpetPainter(depth=2), Transform.create(scale=1), palette),
    ]
    frames = [frames[0], frames[0], frames[1], frames[1]]
    output_file = tmp_path / "video.mp4"

    with patch.object(renderer.image_renderer, "render", wraps=renderer.image_renderer.render) as render:
        renderer.render_video(frames, str(output_file))

    mp4_file = output_file
    assert mp4_file.stat().st_size > 0
    assert render.call_count == 2
    clip = VideoFileClip(str(mp4_file))
    try:
        decoded_frames = list(clip.iter_frames(fps=1))
        assert clip.size == [8, 8]
        assert len(decoded_frames) == 4
    finally:
        clip.close()
