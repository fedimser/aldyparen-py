from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER
from typing import List

import numpy as np

from aldyparen.graphics import Frame, ColorPalette, Transform
from aldyparen.math.hpn import Hpn
from aldyparen.painters import MandelbroidPainter, JuliaPainter, Painter, MandelbrotHighPrecisionPainter, \
    MandelbroidHighPrecisionPainter


def make_animation(frame1: Frame, frame2: Frame, length: int) -> List[Frame]:
    """Continuously transforms frame1 to frame2.
    Returns list of `length+1` frames, where first is frame1, last is frame2.
    """
    assert frame1.painter.__class__ == frame2.painter.__class__
    return [frame1] + [mix_frames(frame1, frame2, i / length) for i in range(1, length)] + [frame2]


def mix_frames(frame1: Frame, frame2: Frame, w: float) -> Frame:
    assert 0 < w < 1
    return Frame(
        painter=mix_painters(frame1.painter, frame2.painter, w),
        transform=mix_transforms(frame1.transform, frame2.transform, w),
        palette=mix_palettes(frame1.palette, frame2.palette, w)
    )


def mix_painters(p1: 'Painter', p2: 'Painter', w: float) -> Painter:
    if p1 == p2:
        return p1
    cl = p1.__class__
    assert p2.__class__ == cl
    if cl == MandelbroidPainter:
        return mix_mandelbroid(p1, p2, w)
    if cl == JuliaPainter:
        return mix_julia(p1, p2, w)
    if cl == MandelbrotHighPrecisionPainter:
        return mix_mandelbrot_hp(p1, p2, w)
    else:
        if p1 != p2:
            raise ValueError("Cannot mix painters.")
        return p1


def mix_transforms(x: Transform, y: Transform, w: float) -> Transform:
    return Transform(
        center_x=mix_hpn(x.center_x, y.center_x, w),
        center_y=mix_hpn(x.center_y, y.center_y, w),
        scale_log10=(1 - w) * x.scale_log10 + w * y.scale_log10,
        rotation=(1 - w) * x.rotation + w * y.rotation
    )


def mix_hpn(x: Hpn, y: Hpn, w: float) -> Hpn:
    return x + (y - x) * w


def extend_palette(x: np.ndarray, new_len: int) -> np.ndarray:
    assert new_len > x.shape[0]
    x = np.tile(x, (new_len // x.shape[0] + 1, 1))
    return x[:new_len, :]


def mix_palettes(x: ColorPalette, y: ColorPalette, w: float) -> ColorPalette:
    c1 = x.colors
    c2 = y.colors
    if c1.shape[0] < c2.shape[0]:
        c1 = extend_palette(c1, c2.shape[0])
    elif c1.shape[0] > c2.shape[0]:
        c2 = extend_palette(c2, c1.shape[0])
    assert c1.shape == c2.shape
    return ColorPalette(
        colors=np.array(np.round((1 - w) * c1 + w * c2), dtype=np.uint8))


def mix_mandelbroid(p1: MandelbroidPainter, p2: MandelbroidPainter, w: float) -> MandelbroidPainter:
    gen_function = mix_functions(p1.gen_function, p2.gen_function, w)
    radius = (1 - w) * p1.radius + w * p2.radius
    max_iter = int(np.round((1 - w) * p1.max_iter + w * p2.max_iter))
    return MandelbroidPainter(gen_function, max_iter=max_iter, radius=radius)


def mix_julia(p1: JuliaPainter, p2: JuliaPainter, w: float) -> JuliaPainter:
    func = mix_functions(p1.func, p2.func, w)
    iters = int(np.round((1 - w) * p1.iters + w * p2.iters))
    tolerance = (1 - w) * p1.tolerance + w * p2.tolerance
    max_colors = int(np.round((1 - w) * p1.max_colors + w * p2.max_colors))
    return JuliaPainter(func=func, iters=iters, tolerance=tolerance, max_colors=max_colors)


def mix_functions(f1: str, f2: str, w: float):
    if f1 == f2:
        return f1
    tokens1 = list(tokenize(BytesIO(f1.encode('utf-8')).readline))
    tokens2 = list(tokenize(BytesIO(f2.encode('utf-8')).readline))
    n = len(tokens1)
    if n != len(tokens2):
        raise ValueError("Functions have different number of tokens")
    result = []
    for i in range(n):
        type1 = tokens1[i].type
        type2 = tokens2[i].type
        val1 = tokens1[i].string
        val2 = tokens2[i].string
        if type1 != type2:
            raise ValueError("Incompatible tokens: %s %s" % (val1, val2))
        if type1 == NUMBER:
            new_val = (1 - w) * float(val1) + w * float(val2)
            new_val_str = str(new_val)
            if 'e' in new_val_str:
                new_val_str = "%.10f" % new_val
            result.append((NUMBER, new_val_str))
        else:
            # Non-number tokens must be identical
            if val1 == val2:
                result.append((type1, val1))
            else:
                raise ValueError("Incompatible tokens: %s %s" % (val1, val2))
    return untokenize(result).decode('utf-8').replace(' ', '')


def mix_mandelbrot_hp(p1: MandelbrotHighPrecisionPainter, p2: MandelbrotHighPrecisionPainter,
                      w: float) -> MandelbrotHighPrecisionPainter:
    max_iter = int(np.round((1 - w) * p1.max_iter + w * p2.max_iter))
    return MandelbrotHighPrecisionPainter(max_iter=max_iter)
