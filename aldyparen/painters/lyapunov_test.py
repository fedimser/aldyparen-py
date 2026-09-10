import os
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from aldyparen.graphics import (
    ChunkingRenderer,
    ColorPalette,
    Frame,
    StaticRenderer,
    Transform,
)
from aldyparen.gui.presets import load_preset
from aldyparen.painters import (
    LyapunovFractalPainter,
    lyapunov,
)
from aldyparen.test_util import _assert_picture


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence": ""}, "sequence must be a non-empty string"),
        ({"sequence": "ABC"}, "sequence must contain only A and B"),
        ({"warmup": -1}, "warmup must be a non-negative integer"),
        ({"iterations": 0}, "iterations must be a positive integer"),
        ({"color_scale": float("inf")}, "color_scale must be a positive finite number"),
    ],
)
def test_lyapunov_rejects_invalid_parameters(kwargs: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        LyapunovFractalPainter(**kwargs)


def test_lyapunov_paints_known_stable_chaotic_and_superstable_points():
    painter = LyapunovFractalPainter(sequence="A", warmup=100, iterations=100, color_scale=10)
    points = np.array([3 + 3j, 4 + 4j, 2 + 2j], dtype=np.complex128)
    ans = np.zeros(3, dtype=np.uint32)

    painter.paint(points, ans)

    np.testing.assert_array_equal(ans, [1, 28, np.iinfo(np.uint32).max])
    assert painter.warning is None


def test_lyapunov_paint_warns_and_fills_non_finite_points():
    painter = LyapunovFractalPainter(sequence="AB", warmup=1, iterations=1)
    ans = np.full(2, 99, dtype=np.uint32)

    painter.paint(np.array([complex(np.nan, 1), complex(1, np.inf)]), ans)

    np.testing.assert_array_equal(ans, [0, 0])
    assert painter.warning == "Could not compute Lyapunov exponents for 2 point(s)."


def test_lyapunov_paint_handles_runtime_errors(monkeypatch: pytest.MonkeyPatch):
    painter = LyapunovFractalPainter()
    ans = np.full(2, 99, dtype=np.uint32)

    def fail(*_args: object):
        raise RuntimeError("calculation failed")

    monkeypatch.setattr(lyapunov, "is_cuda_available", lambda: False)
    monkeypatch.setattr(lyapunov, "paint_lyapunov_numba", fail)
    painter.paint(np.array([0j, 1j]), ans)

    np.testing.assert_array_equal(ans, [0, 0])
    assert painter.warning == "Error computing Lyapunov exponents: calculation failed"


def test_lyapunov_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch):
    painter = LyapunovFractalPainter()
    points = np.array([0j, 1j], dtype=np.complex128)
    ans = np.zeros(2, dtype=np.uint32)

    def paint_cuda(
        actual_points: NDArray[np.complex128],
        actual_ans: NDArray[np.uint32],
        *_args: object,
    ) -> int:
        assert actual_points is points
        actual_ans[:] = [7, 8]
        return 0

    monkeypatch.setattr(lyapunov, "is_cuda_available", lambda: True)
    monkeypatch.setattr(lyapunov, "paint_lyapunov_cuda", paint_cuda)
    monkeypatch.setattr(lyapunov, "paint_lyapunov_numba", lambda *_args: pytest.fail("CPU fallback used"))

    painter.paint(points, ans)

    np.testing.assert_array_equal(ans, [7, 8])
    assert painter.warning is None


def test_lyapunov_falls_back_to_cpu_when_cuda_fails(monkeypatch: pytest.MonkeyPatch):
    painter = LyapunovFractalPainter()
    ans = np.zeros(2, dtype=np.uint32)

    def fail_cuda(*_args: object):
        raise RuntimeError("CUDA failed")

    def paint_cpu(_points: object, actual_ans: NDArray[np.uint32], *_args: object) -> int:
        actual_ans[:] = [3, 4]
        return 0

    monkeypatch.setattr(lyapunov, "is_cuda_available", lambda: True)
    monkeypatch.setattr(lyapunov, "paint_lyapunov_cuda", fail_cuda)
    monkeypatch.setattr(lyapunov, "paint_lyapunov_numba", paint_cpu)

    painter.paint(np.array([0j, 1j]), ans)

    np.testing.assert_array_equal(ans, [3, 4])
    assert painter.warning is None


@pytest.mark.cuda_sim
@pytest.mark.skipif(
    os.environ.get("NUMBA_ENABLE_CUDASIM") != "1",
    reason="requires NUMBA_ENABLE_CUDASIM=1 before importing numba.cuda",
)
def test_lyapunov_cuda_simulator_matches_cpu_kernel():
    values = np.linspace(2.0, 4.0, 257)
    points = (values + 1j * values[::-1]).astype(np.complex128)
    points[:5] = [3 + 3j, 4 + 4j, 2 + 2j, complex(np.nan, 1), complex(1, np.inf)]
    sequence = np.array([0, 0, 1, 0, 1], dtype=np.uint8)
    cpu_ans = np.empty(len(points), dtype=np.uint32)
    cuda_ans = np.empty(len(points), dtype=np.uint32)

    cpu_failures = lyapunov.paint_lyapunov_numba(points, cpu_ans, sequence, 20, 30, 10.0)
    cuda_failures = lyapunov.paint_lyapunov_cuda(points, cuda_ans, sequence, 20, 30, 10.0)

    np.testing.assert_array_equal(cuda_ans, cpu_ans)
    assert cuda_failures == cpu_failures


def test_lyapunov_chunked_render_matches_static_render():
    frame = Frame(
        LyapunovFractalPainter(sequence="AABAB", warmup=20, iterations=30),
        Transform.create(center=3 + 3j, scale=2),
        ColorPalette.gradient("black", "white", size=32),
    )

    static_picture = StaticRenderer(20, 15).render(frame)
    chunked_picture = ChunkingRenderer(20, 15, chunk_size=17).render(frame)

    np.testing.assert_array_equal(chunked_picture, static_picture)
