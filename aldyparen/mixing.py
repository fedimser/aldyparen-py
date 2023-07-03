from aldyparen.graphics import Frame, ColorPalette, Transform
from typing import List
import numpy as np


def make_animation(frame1: Frame, frame2: Frame, length: int) -> List[Frame]:
    """Continuously transforms frame1 to frame2.
    Returns list of `length+1` frames, where first is frame1, last is frame2.
    """
    assert frame1.painter.__class__ == frame2.painter.__class__
    assert frame1.palette.colors.shape == frame2.palette.colors.shape
    return [frame1] + [mix_frames(frame1, frame2, i / length) for i in range(1, length)] + [frame2]


def mix_frames(frame1: Frame, frame2: Frame, w: float) -> Frame:
    assert 0 < w < 1
    return Frame(
        painter=mix_painters(frame1.painter, frame2.painter, w),
        transform=mix_transforms(frame1.transform, frame2.transform, w),
        palette=mix_palettes(frame1.palette, frame2.palette, w)
    )


def mix_painters(p1: 'Painter', p2: 'Painter', w: float) -> Frame:
    if p1 == p2:
        return p1
    # TODO: implement.
    return p1


def mix_transforms(x: Transform, y: Transform, w: float) -> Transform:
    return Transform(
        center=np.complex128((1 - w) * x.center + w * y.center),
        scale=np.exp((1 - w) * np.log(x.scale) + w * np.log(y.scale)),
        rotation=(1 - w) * x.rotation + w * y.rotation
    )


def mix_palettes(x: ColorPalette, y: ColorPalette, w: float) -> ColorPalette:
    return ColorPalette(
        colors=np.array(np.round((1 - w) * x.colors + w * y.colors), dtype=np.uint8))
