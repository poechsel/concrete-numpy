"""
Tests of execution of add operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp

from concrete.numpy.dtypes.integer import SignedInteger


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x == y,
            id="x == y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 15], "status": "encrypted"},
            "y": {"range": [0, 15], "status": "encrypted"},
        },
    ],
)
def _test_equal(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x != y,
            id="x != y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 15], "status": "encrypted"},
            "y": {"range": [0, 15], "status": "encrypted"},
        },
    ],
)
def _test_not_equal(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: x < 0,
            id="x < 0",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [-7, 0], "status": "encrypted"},
        },
    ],
)
def _test_less_2(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    print("sample", sample)

    helpers.check_execution(circuit, function, [-7])


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x < y,
            id="x < y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [-2, 1], "status": "encrypted"},
            "y": {"range": [-2, 1], "status": "encrypted"},
        },
    ],
)
def _test_less(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    print("sample", sample)

    helpers.check_execution(circuit, function, [-2, -1])
    return
    for i in range(-2, 1):
        for j in range(-2, 1):
            sample = [i, j]
            print(i, j)
            helpers.check_execution(circuit, function, sample)


def test_less_pattern():
    """
    Test add where both of the operators are dynamic.
    """
    width = 8
    high = (1 << (width - 1)) - 1
    low = -(1 << (width - 1))
    for x in range(low, high + 1):
        for y in range(low, high + 1):
            x = np.int32(x)
            y = np.int32(y)
            
            assert (
                cnp.mlir.node_rewriter.less(
                    SignedInteger(width),
                    SignedInteger(width),
                    x + (1 << width) if x < 0 else x,
                    y + (1 << width) if y < 0 else y,
                )
                == (x < y)
            )


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x <= y,
            id="x <= y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [-7, 7], "status": "encrypted"},
            "y": {"range": [-7, 7], "status": "encrypted"},
        },
    ],
)
def _test_less_equal(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x > y,
            id="x > y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [-7, 7], "status": "encrypted"},
            "y": {"range": [-7, 7], "status": "encrypted"},
        },
    ],
)
def _test_greater(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x >= y,
            id="x >= y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [-7, 7], "status": "encrypted"},
            "y": {"range": [-7, 7], "status": "encrypted"},
        },
    ],
)
def _test_greater_equal(function, parameters, helpers):
    """
    Test add where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
