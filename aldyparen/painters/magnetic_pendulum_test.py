from typing import Any

import numpy as np
import pytest

from aldyparen.graphics import ChunkingRenderer, Frame, StaticRenderer
from aldyparen.gui.presets import load_preset
from aldyparen.painters import MagneticPendulumPainter, magnetic_pendulum
from aldyparen.test_util import _assert_picture


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"magnets": []}, "magnets must be a list containing at least three magnets"),
        (
            {
                "magnets": [
                    {"x": 0, "y": 0, "strength": 0},
                    {"x": 1, "y": 0, "strength": 1},
                    {"x": 0, "y": 1, "strength": 1},
                ]
            },
            "magnet 0 strength must be positive",
        ),
        ({"height": 0}, "height must be a positive finite number"),
        ({"damping": -1}, "damping must be a non-negative finite number"),
        ({"gravity": float("inf")}, "gravity must be a non-negative finite number"),
        ({"time_step": 0}, "time_step must be a positive finite number"),
        ({"max_steps": 0}, "max_steps must be a positive integer"),
        ({"settle_distance": 0}, "settle_distance must be a positive finite number"),
        ({"settle_speed": -1}, "settle_speed must be a non-negative finite number"),
    ],
)
def test_magnetic_pendulum_rejects_invalid_parameters(kwargs: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        MagneticPendulumPainter(**kwargs)


def test_magnetic_pendulum_paints_known_magnets_and_step_limit():
    magnets = [
        {"x": -1, "y": 0, "strength": 1},
        {"x": 1, "y": 0, "strength": 1},
        {"x": 0, "y": 1, "strength": 1},
    ]
    painter = MagneticPendulumPainter(magnets=magnets, max_steps=1, settle_distance=0.01)
    points = np.array([-1 + 0j, 1 + 0j, 1j, 10 + 10j], dtype=np.complex128)
    ans = np.full(4, 99, dtype=np.uint32)

    painter.paint(points, ans)

    np.testing.assert_array_equal(ans, [1, 2, 3, 0])
    assert painter.warning is None


def test_magnetic_pendulum_paint_warns_and_fills_non_finite_points():
    painter = MagneticPendulumPainter(max_steps=1)
    ans = np.full(2, 99, dtype=np.uint32)

    painter.paint(np.array([complex(np.nan, 1), complex(1, np.inf)]), ans)

    np.testing.assert_array_equal(ans, [0, 0])
    assert painter.warning == "Could not simulate magnetic pendulum for 2 point(s)."


def test_magnetic_pendulum_paint_handles_runtime_errors(monkeypatch: pytest.MonkeyPatch):
    painter = MagneticPendulumPainter()
    ans = np.full(2, 99, dtype=np.uint32)

    def fail(*_args: object):
        raise RuntimeError("calculation failed")

    monkeypatch.setattr(magnetic_pendulum, "paint_magnetic_pendulum_numba", fail)
    painter.paint(np.array([0j, 1j]), ans)

    np.testing.assert_array_equal(ans, [0, 0])
    assert painter.warning == "Error simulating magnetic pendulum: calculation failed"


def test_magnetic_pendulum_chunked_render_matches_static_render():
    painter, transform, palette = load_preset("magnetic_pendulum")
    assert isinstance(painter, MagneticPendulumPainter)
    painter.max_steps = 20
    frame = Frame(painter, transform, palette)

    static_picture = StaticRenderer(20, 15).render(frame)
    chunked_picture = ChunkingRenderer(20, 15, chunk_size=17).render(frame)

    np.testing.assert_array_equal(chunked_picture, static_picture)
