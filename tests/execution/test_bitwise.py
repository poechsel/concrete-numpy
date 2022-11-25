"""
Tests of execution of add operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x | y,
            id="x | y",
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
def test_bitwise_or(function, parameters, helpers):
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
            lambda x, y: x & y,
            id="x & y",
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
def test_bitwise_and(function, parameters, helpers):
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
            lambda x, y: x ^ y,
            id="x ^ y",
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
def test_bitwise_xor(function, parameters, helpers):
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
            lambda x: ~x,
            id="~x",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [7, 15], "status": "encrypted"},
        },
    ],
)
def test_bitwise_not(function, parameters, helpers):
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
