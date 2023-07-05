import json

from aldyparen import ALL_PAINTERS, MandelbroidPainter
from aldyparen.util import SUPPORTED_FUNCTIONS


def test_defaults():
    for painter_class in ALL_PAINTERS:
        painter = painter_class()
        config1 = painter.to_object()
        config1_json = json.dumps(config1)
        assert config1_json == json.dumps(json.loads(config1_json))
        painter2 = painter_class(**config1)
        config2 = painter2.to_object()
        assert config1_json == json.dumps(config2)


def test_mandelbroid_rejects_bad_functions():
    error_msg = None
    try:
        MandelbroidPainter(gen_function="f(c,z)")
    except ValueError as err:
        error_msg = str(err)
    assert "Unexpected token: f" in error_msg


def test_mandelbroid_supports_all_functions():
    gen_function = '+'.join(func + '(c+z+1j)' for func in SUPPORTED_FUNCTIONS)
    MandelbroidPainter(gen_function=gen_function)
