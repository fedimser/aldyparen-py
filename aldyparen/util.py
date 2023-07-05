import re

import numpy as np

SUPPORTED_FUNCTIONS = {
    "exp", "log", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "real", "imag",
    "abs", "angle"
}
OK_TOKEN_REGEX = re.compile(r"[+\-*/().0123456789]+")


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
        elif token in ok_tokens:
            result += token
        elif OK_TOKEN_REGEX.match(token):
            result += token
        else:
            raise ValueError(f"Unexpected token: {token}")

    # Check that this is valid mathematical function by evaluating it.
    env = {var: np.complex128(1) for var in variables}
    env["np"] = np
    test_value = eval(result, env)
    if type(test_value) != np.complex128:
        raise ValueError(f"Not a valid function")

    return result
