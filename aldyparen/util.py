import re

import numpy as np

SUPPORTED_FUNCTIONS = {
    "exp", "log", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "real", "imag",
    "abs", "angle"
}


def prepare_function(function, variables=[]):
    """Validates and prepares function for evaluation with numpy."""
    assert function.isascii(), "Bad character"
    tokens = re.findall('[a-zA-Z]+|[^a-zA-Z]+', function)
    result = ''
    ok_tokens = set(variables)
    ok_tokens.add('j')
    for token in tokens:
        if token in SUPPORTED_FUNCTIONS:
            result += 'np.' + token
        elif not token.isalpha() or token in ok_tokens:
            result += token
        else:
            raise ValueError(f"Unexpected token: {token}")
    eval(result, {"np": np, "c": 1, "z": 1})
    return result
