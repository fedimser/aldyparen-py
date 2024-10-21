import numpy as np

from aldyparen.math.complex_hpn import ComplexHpn, is_on_or_outside_circle
import random

from aldyparen.math.hpn import Hpn


def test_conversion():
    for x in [0, 1, 2.3 + 4.5j, 100 - 10j]:
        x_hpn = ComplexHpn.from_number(x)
        x2 = x_hpn.approx
        assert np.allclose(x, x2)


def test_arithmetic():
    for _ in range(10):
        a = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        b = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        a_hpn = ComplexHpn.from_number(a)
        b_hpn = ComplexHpn.from_number(b)
        assert np.isclose((a_hpn + b_hpn).approx, a + b)
        assert np.isclose((a_hpn - b_hpn).approx, a - b)
        assert np.isclose((a_hpn * b_hpn).approx, a * b)


def test_is_on_or_outside_circle():
    for _ in range(10):
        z = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        z_raw = ComplexHpn.from_number(z).to_raw()
        r = np.abs(z)
        assert is_on_or_outside_circle(z_raw, Hpn.from_number(r ** 2 - 1e-5).digits) is True
        assert is_on_or_outside_circle(z_raw, Hpn.from_number(r ** 2 + 1e-5).digits) is False
