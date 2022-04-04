"""
Tests of `CompilationConfiguration` class.
"""

import pytest

from concrete.numpy.compilation import CompilationConfiguration


@pytest.mark.parametrize(
    "kwargs,expected_error,expected_message",
    [
        pytest.param(
            {"enable_unsafe_features": False, "use_insecure_key_cache": True},
            RuntimeError,
            "Insecure key cache cannot be used without enabling unsafe features",
        ),
    ],
)
def test_configuration_bad_init(kwargs, expected_error, expected_message):
    """
    Test `__init__` method of `CompilationConfiguration` class with bad parameters.
    """

    with pytest.raises(expected_error) as excinfo:
        CompilationConfiguration(**kwargs)

    assert str(excinfo.value) == expected_message