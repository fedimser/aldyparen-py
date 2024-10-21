import string
from typing import Callable, Optional, Any

import numba

from aldyparen.math import complex_hpn
from aldyparen.math.complex_hpn import ComplexHpn
from aldyparen.math.hpn import DEFAULT_PRECISION
from aldyparen.util import prepare_eval_env

ALLOWED_CHARS = set(string.ascii_lowercase + string.digits + ".,()+-*")
SUPPORTED_UNARY_FUNCTIONS = ["sqr", "abscw"]


def _ce(value: Any) -> "CompilerExpression":
    if type(value) is CompilerExpression:
        return value
    return CompilerExpression(constant=value)


class CompilerExpression:
    def __init__(self,
                 variable: Optional[str] = None,
                 constant: Optional[Any] = None,
                 function: Optional[tuple[str, list["CompilerExpression"]]] = None):
        assert int(variable is not None) + int(constant is not None) + int(
            function is not None) == 1
        self.variable = variable
        self.constant = constant
        self.function = function

    def __add__(self, other):
        return CompilerExpression(function=("complex_hpn.add", [self, _ce(other)]))

    def __radd__(self, other):
        return CompilerExpression(function=("complex_hpn.add", [_ce(other), self]))

    def __sub__(self, other):
        return CompilerExpression(function=("complex_hpn.sub", [self, _ce(other)]))

    def __rsub__(self, other):
        return CompilerExpression(function=("complex_hpn.sub", [_ce(other), self]))

    def __mul__(self, other):
        return CompilerExpression(function=("complex_hpn.mul", [self, _ce(other)]))

    def __rmul__(self, other):
        return CompilerExpression(function=("complex_hpn.mul", [_ce(other), self]))

    def __pow__(self, other):
        if type(other) is int:
            assert other >= 1, "Only positive integer power is supported"
            if other == 1:
                return self
            elif other == 2:
                return CompilerExpression(function=("complex_hpn.sqr", [self]))
            else:
                return CompilerExpression(function=("power_int", [self, other]))
        raise f"Power of {other} is not supported"

    def to_hpn_expression(self, constants: dict[str, ComplexHpn], prec: int) -> str:
        if self.variable is not None:
            return self.variable
        if self.constant is not None:
            const_name = "__const" + str(len(constants))
            constants[const_name] = ComplexHpn.from_number(self.constant, prec=prec)
            return const_name
        assert self.function is not None
        func_name, args = self.function
        if func_name == "power_int":
            arg0 = args[0].to_hpn_expression(constants, prec)
            assert type(args[1]) is int
            assert args[1] >= 3  # type:ignore
            return f"complex_hpn.power_int({arg0},{args[1]})"
        args = ','.join(expr.to_hpn_expression(constants, prec) for expr in args)
        return f"{func_name}({args})"


# Returns function that evaluates operates on raw HPCNs (i.e. pairs of 1D arrays).
# Resulting function takes N values of type (i8[:],i8[:]) and returns Tuple (i8[:],i8[:]).
def compile_expression_hpcn(expr: str, var_names: list[str], precision=DEFAULT_PRECISION) -> Callable:
    for char in expr:
        assert char in ALLOWED_CHARS, f"Invalid character {char}"

    eval_env = prepare_eval_env()
    for var_name in var_names:
        eval_env[var_name] = CompilerExpression(variable=var_name)
    for unary_func in SUPPORTED_UNARY_FUNCTIONS:
        func_full_name = f"complex_hpn.{unary_func}"
        eval_env[unary_func] = lambda x, f=func_full_name: CompilerExpression(function=(f, [x]))

    try:
        compiled_expr = eval(expr, eval_env)  # type: CompilerExpression
    except TypeError:
        raise ValueError("Invalid expression (maybe used used unknown function or variable).")
    assert type(compiled_expr) is CompilerExpression
    constants = dict()  # type: dict[str, ComplexHpn]
    numba_expr = compiled_expr.to_hpn_expression(constants, precision)
    numba_args = ",".join(var_names)
    numba_ret_type = "UniTuple(i8[:],2)"
    numba_arg_types = ",".join([numba_ret_type for _ in range(len(var_names))])
    numba_signature = f"{numba_ret_type}({numba_arg_types})"

    numba_env = {
        "numba": numba,
        "complex_hpn": complex_hpn,
    }
    numba_source = ""
    for const_name, const_value in constants.items():
        numba_env[const_name] = const_value.to_raw()
        numba_source += f"{const_name}={const_name}\n"
    numba_source += "\n".join([
        f'@numba.jit("{numba_signature}",nopython=True)',
        f'def __func({numba_args}):',
        f'  return {numba_expr}',
    ])
    exec(numba_source, numba_env)
    return numba_env["__func"]  # type: ignore
