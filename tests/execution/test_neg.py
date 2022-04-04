"""
Tests of execution of neg operation.
"""

import numpy as np
import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 64], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 64], "status": "encrypted", "shape": (3, 2)},
        },
    ],
)
def test_neg(parameters, helpers):
    """
    Test neg.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    @cnp.compiler(parameter_encryption_statuses, configuration=configuration)
    def operator(x):
        return -x

    @cnp.compiler(parameter_encryption_statuses, configuration=configuration)
    def function(x):
        return np.negative(x)

    inputset = helpers.generate_inputset(parameters)

    operator_circuit = operator.compile(inputset)
    function_circuit = function.compile(inputset)

    sample = helpers.generate_sample(parameters)

    helpers.check_execution(operator_circuit, operator, sample)
    helpers.check_execution(function_circuit, function, sample)
