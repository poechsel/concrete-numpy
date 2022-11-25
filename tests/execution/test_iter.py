"""
Tests of execution of iteration of tracer.
"""

import numpy as np
import pytest

import concrete.numpy as cnp
from concrete.numpy.dtypes import UnsignedInteger
from concrete.numpy.tracing.tracer import Tracer
from concrete.numpy.values import EncryptedScalar


@pytest.mark.parametrize("shape", [(3,), (3, 2), (3, 2, 4)])
def test_iter(shape, helpers):
    """
    Test iteration of tracers.
    """

    def function(x):
        result = cnp.zeros(x.shape[1:])
        for value in x:
            result += value
        return result

    configuration = helpers.configuration()
    compiler = cnp.Compiler(function, {"x": "encrypted"})

    inputset = [np.random.randint(0, 2**2, size=shape) for _ in range(100)]
    circuit = compiler.compile(inputset, configuration)

    sample = np.random.randint(0, 2**2, size=shape)
    helpers.check_execution(circuit, function, sample)


def test_iter2(helpers):
    """
    Test iteration of tracers.
    """

    def right_shift(x, y):
        result = x
        for value in range(16):
            result = result >> (y <= value)
        return result

    graph = Tracer.trace(
        right_shift,
        {
            "x": EncryptedScalar(UnsignedInteger(8)),
            "y": EncryptedScalar(UnsignedInteger(4)),
        },
    )
    print(graph.input_nodes)
    print(graph.output_nodes)
    # configuration = helpers.configuration()
    # compiler = cnp.Compiler(function, {"x": "encrypted"})
    # print(compiler)
    # inputset = [np.random.randint(0, 2 ** 2) for _ in range(100)]
    # circuit = compiler.compile(inputset, configuration)
    # print(circuit)
    # sample = np.random.randint(0, 2 ** 2)
    # helpers.check_execution(circuit, function, sample)
