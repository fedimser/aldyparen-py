from aldyparen.math.complex_hpn import ComplexHpn
from aldyparen.math.hpn_compiler import compile_expression_hpcn

import numpy as np
import random
from typing import Callable


def check_univariate(expr: str, golden: Callable[[complex], complex]):
    evaluator = compile_expression_hpcn(expr, ["z"])
    for _ in range(10):
        z = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        ans = ComplexHpn.from_raw(evaluator(ComplexHpn.from_number(z).to_raw())).to_complex()
        expected = np.complex128(golden(z))
        assert np.allclose(ans, expected)


def test_univariate():
    check_univariate("z", lambda z: z)
    check_univariate("z*2", lambda z: z * 2)
    check_univariate("z*3.14", lambda z: z * 3.14)
    check_univariate("z*(3.14+1j)", lambda z: z * (3.14 + 1j))
    check_univariate("2*z", lambda z: 2 * z)
    check_univariate("z*z", lambda z: z * z)
    check_univariate("z**2", lambda z: z ** 2)
    check_univariate("sqr(z)", lambda z: z ** 2)
    check_univariate("z*z*z", lambda z: z ** 3)
    check_univariate("z+5", lambda z: z + 5)
    check_univariate("z+z", lambda z: z + z)
    check_univariate("(z+1)*(z+2)*(z+3)", lambda z: (z + 1) * (z + 2) * (z + 3))
    check_univariate("z-5", lambda z: z - 5)
    check_univariate("(z-1)*(z+2)*(3-z)*(4j+z)", lambda z: (z - 1) * (z + 2) * (3 - z) * (4j + z))


def check_bivariate(expr: str, golden: Callable[[complex, complex], complex]):
    evaluator = compile_expression_hpcn(expr, ["a", "b"])
    for _ in range(10):
        a = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        b = random.uniform(-100, 100) + 1j * random.uniform(-100, 100)
        ans = ComplexHpn.from_raw(evaluator(ComplexHpn.from_number(a).to_raw(),
                                            ComplexHpn.from_number(b).to_raw())).to_complex()
        expected = np.complex128(golden(a, b))
        assert np.allclose(ans, expected)


def test_bivariate():
    check_bivariate("a+b", lambda a, b: a + b)
    check_bivariate("a-b", lambda a, b: a - b)
    check_bivariate("a*b", lambda a, b: a * b)
    check_bivariate("a", lambda a, b: a)
    check_bivariate("(a**2*b)+(2.0*a*3.0*b**2)+a",
                    lambda a, b: (a ** 2 * b) + (2.0 * a * 3.0 * b ** 2) + a)
    # Burning ship.
    check_bivariate("abscw(a)**2+b", lambda a, b: (abs(a.real) + 1j * abs(a.imag)) ** 2 + b)
