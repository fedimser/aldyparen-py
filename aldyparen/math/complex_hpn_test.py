import numpy as np

from aldyparen.math.complex_hpn import ComplexHpn, mul
import random


def test_conversion():
    for x in [0, 1, 2.3 + 4.5j, 100 - 10j]:
        x_hpn = ComplexHpn.from_number(x)
        x2 = x_hpn.to_complex()
        assert np.allclose(x, x2)


def test_arithmetic():
    for _ in range(10):
        a = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        b = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        a_hpn = ComplexHpn.from_number(a)
        b_hpn = ComplexHpn.from_number(b)
        assert np.isclose((a_hpn + b_hpn).to_complex(), a + b)
        assert np.isclose((a_hpn - b_hpn).to_complex(), a - b)
        assert np.isclose((a_hpn * b_hpn).to_complex(), a * b)
