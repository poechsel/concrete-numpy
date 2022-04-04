"""
Tests of execution of dot operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "size",
    [1, 4, 6, 10],
)
def test_dot(size, helpers):
    """
    Test dot.
    """

    configuration = helpers.configuration()

    bound = int(np.floor(np.sqrt(127 / size)))
    cst = np.random.randint(0, bound, size=(size,))

    @cnp.compiler({"x": "encrypted"}, configuration=configuration)
    def left_function(x):
        return np.dot(x, cst)

    @cnp.compiler({"x": "encrypted"}, configuration=configuration)
    def right_function(x):
        return np.dot(cst, x)

    @cnp.compiler({"x": "encrypted"}, configuration=configuration)
    def method(x):
        return x.dot(cst)

    inputset = [np.random.randint(0, bound, size=(size,)) for i in range(100)]

    left_function_circuit = left_function.compile(inputset)
    right_function_circuit = right_function.compile(inputset)
    method_circuit = method.compile(inputset)

    sample = np.random.randint(0, bound, size=(size,), dtype=np.uint8)

    helpers.check_execution(left_function_circuit, left_function, sample)
    helpers.check_execution(right_function_circuit, right_function, sample)
    helpers.check_execution(method_circuit, method, sample)
