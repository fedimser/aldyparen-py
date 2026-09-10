from unittest.mock import Mock

import pytest

from aldyparen.graphics import Frame
from aldyparen.gui.app import AldyparenApp
from aldyparen.gui.async_runners import MoviePreviewRenderRunnable


@pytest.mark.parametrize(
    ("frames_after_render", "selected_frame_idx_after_render"),
    [([], -1), ([Frame.default()], -1), ([Frame.default()], 0)],
)
def test_movie_preview_completion_ignores_invalid_selection(
    frames_after_render: list[Frame], selected_frame_idx_after_render: int
):
    frame = Frame.default()
    app = AldyparenApp.__new__(AldyparenApp)
    app.frames = [frame]
    app.selected_frame_idx = 0
    app.shown_movie_frame_is_invalid = False
    app.movie_frame_renderer = Mock()

    def change_movie_during_render(_: Frame):
        app.frames = frames_after_render
        app.selected_frame_idx = selected_frame_idx_after_render
        return "preview"

    app.movie_frame_renderer.render.side_effect = change_movie_during_render

    MoviePreviewRenderRunnable(app, frame).run()

    assert frame.cached_movie_preview == "preview"
    assert not app.shown_movie_frame_is_invalid