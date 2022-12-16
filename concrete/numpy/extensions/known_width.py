"""
Declaration of `univariate` function.
"""

from typing import Any, Callable, Union

import numpy as np

from ..dtypes import Float
from ..representation import Node
from ..tracing import Tracer
from ..values import Value
from ..dtypes import Float, Integer


def known_width(values: Any, bit_width: int, force_unsigned=False) -> Any:
    """
    Wrap a univariate function so that it is traced into a single generic node.

    Args:
        function (Callable[[Any], Any]):
            univariate function to wrap

    Returns:
        Callable[[Union[Tracer, Any]], Union[Tracer, Any]]:
            another univariate function that can be called with a Tracer as well
    """

    if isinstance(values, int):
        if force_unsigned:
            return (values + (1 << 32)) & ((1 << bit_width) - 1)
        else:
            return values & ((1 << bit_width) - 1)
    # pylint: disable=protected-access
    is_tracing = Tracer._is_tracing
    # pylint: enable=protected-access

    if not is_tracing:
        return values
    print(values, values.shape)
    print(values.computation.format(predecessors=["a", "b"]))
    print("output", values.output, bit_width)
    if isinstance(values.output.dtype, Integer):
        values.output.dtype.bit_width = bit_width
        if force_unsigned:
            values.output.dtype.is_signed = False
    elif isinstance(values.output.dtype, Float):
        values.output.dtype.bit_width = bit_width
    return values
