"""
Declaration of `GraphConverter` class.
"""

# pylint: disable=no-member,no-name-in-module

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import concrete.lang as concretelang
import networkx as nx
import numpy as np
from concrete.lang.dialects import fhe, fhelinalg
from mlir.dialects import arith, func
from mlir.ir import (
    Context,
    DenseElementsAttr,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    OpResult,
    RankedTensorType,
)

from ..dtypes import BaseDataType, Integer, SignedInteger, UnsignedInteger
from ..extensions.table import LookupTable
from ..internal.utils import assert_that
from ..representation import Graph, Node, Operation
from ..tracing.tracer import Tracer
from ..values import ClearScalar, EncryptedScalar, EncryptedTensor, Value
from .node_converter import NodeConverter
from .utils import MAXIMUM_TLU_BIT_WIDTH

# pylint: enable=no-member,no-name-in-module


class BinaryNodeRewriter:
    """
    GraphConverter class, to convert computation graphs to their MLIR equivalent.
    """

    patterns: Dict[
        str,
        Callable,
    ]
    cache: Dict[Any, Graph]

    def __init__(self):
        self.patterns = {}
        self.cache = {}

    def register(
        self,
        ops: str,
        macro: Callable,
    ):
        self.patterns[ops] = macro

    def register_from_python(
        self,
        ops: str,
        macro: Callable,
    ):
        def from_python(x, y, output):
            f = macro(x.dtype, y.dtype, output.dtype)
            return Tracer.trace(
                f,
                {"x": x, "y": y},
            )

        self.patterns[ops] = from_python

    def find_sub_graph_for(
        self,
        ops: str,
        x: Value,
        y: Value,
        output: Value,
    ) -> Graph:
        assert_that(ops in self.patterns)
        key = (ops, x.dtype, y.dtype, output.dtype)
        if key not in self.cache:
            self.cache[key] = self.patterns[ops](x, y, output)
        return deepcopy(self.cache[key])

    def run(self, graph: Graph):
        """
        Tensorize scalars if they are used within fhelinalg operations.

        Args:
            graph (Graph):
                computation graph to update
        """

        # tensorized_scalars: Dict[Node, Node] = {}

        nx_graph = graph.graph
        all_nodes = list(nx_graph.nodes)
        for node in all_nodes:
            if node.operation == Operation.Generic and node.properties["name"] in self.patterns:
                assert len(node.inputs) == 2

                # This will be left as-is and compiled using a LUT
                if not all(input.is_scalar for input in node.inputs):
                    continue
                sub_graph = self.find_sub_graph_for(
                    node.properties["name"],
                    node.inputs[0],
                    node.inputs[1],
                    node.output,
                )

                print(sub_graph.format())
                graph.expand_node(node, sub_graph)


def bitwise_and(bit_width, x, y):
    table = LookupTable([0, 0, 1])
    z = 0
    for i in range(bit_width):
        lsb_x = (x >> i) & 1
        lsb_y = (y >> i) & 1
        lsb = lsb_x + lsb_y
        z += table[lsb] << i
    return z


def bitwise_xor(bit_width, x, y):
    table = LookupTable([0, 1, 0])
    z = 0
    for i in range(bit_width):
        lsb_x = (x >> i) & 1
        lsb_y = (y >> i) & 1
        lsb = lsb_x + lsb_y
        z += table[lsb] << i
    return z


def bitwise_or(bit_width, x, y):
    table = LookupTable([0, 1, 1])
    z = 0
    for i in range(bit_width):
        lsb_x = (x >> i) & 1
        lsb_y = (y >> i) & 1
        lsb = lsb_x + lsb_y
        z += table[lsb] << i
    return z


def equal(bit_width, x, y):
    return (x - y) == 0


def not_equal(bit_width, x, y):
    return ((x - y) != 0) & 1


def less(bit_width, x, y):
    # return (((x >= 0 && y > 0) || (x < 0 && y <= 0)) && (x - y < 0))
    #        || (x < 0 && y > 0);

    table_or = LookupTable([0, 1, 1])
    table_and = LookupTable([0, 0, 1])
    x_le_0 = (x < 0) & 1
    y_ge_0 = (y > 0) & 1
    and_x_geq_0_and_y_ge_0 = table_and[(x_le_0 ^ 1) + ((y_ge_0))]
    and_x_le_0_and_y_le_0 = table_and[x_le_0 + ((y_ge_0 ^ 1))]
    and_x_le_0_and_y_geq_0 = table_and[(x_le_0 & 1) + ((y_ge_0))]
    dxy_le_0 = (x - y) < 0
    or_same_sign = table_or[and_x_geq_0_and_y_ge_0 + and_x_le_0_and_y_le_0]
    or_1 = table_and[or_same_sign + dxy_le_0]
    return table_or[or_1 + and_x_le_0_and_y_geq_0]


def less_equal(bit_width, x, y):

    # Could be expressed with (x < y) || (x == 0) but it's more economic that way
    # (((x >= 0 && y >= 0) || (x < 0 && y < 0)) && (x - y <= 0))
    #    || (x < 0 && y >= 0);

    table_or = LookupTable([0, 1, 1])
    table_and = LookupTable([0, 0, 1])
    # Either x - y < 0
    # Or )
    x_le_0 = (x < 0) & 1
    y_geq_0 = (y >= 0) & 1
    and_x_geq_0_and_y_ge_0 = table_and[(x_le_0 ^ 1) + ((y_geq_0))]
    and_x_le_0_and_y_le_0 = table_and[x_le_0 + ((y_geq_0 ^ 1))]
    and_x_le_0_and_y_geq_0 = table_and[(x_le_0 & 1) + ((y_geq_0))]
    dxy_le_0 = (x - y) <= 0
    or_same_sign = table_or[and_x_geq_0_and_y_ge_0 + and_x_le_0_and_y_le_0]
    or_1 = table_and[or_same_sign + dxy_le_0]
    return table_or[or_1 + and_x_le_0_and_y_geq_0]


def greater(bit_width, x, y):
    return less_equal(bit_width, y, x)


def greater_equal(bit_width, x, y):
    return less(bit_width, y, x)


def make_int_bin_op(
    callback, x_type: BaseDataType, y_type: BaseDataType, output_type: BaseDataType
) -> Callable[[Any, Any], Any]:
    assert_that(isinstance(x_type, Integer))
    assert_that(isinstance(y_type, Integer))
    return partial(callback, max(x_type.bit_width, y_type.bit_width))


def rewrite_all_binops(graph):
    rewriter = BinaryNodeRewriter()
    rewriter.register_from_python("bitwise_or", partial(make_int_bin_op, bitwise_or))
    rewriter.register_from_python("bitwise_and", partial(make_int_bin_op, bitwise_and))
    rewriter.register_from_python("bitwise_xor", partial(make_int_bin_op, bitwise_xor))
    rewriter.register_from_python("equal", partial(make_int_bin_op, equal))
    rewriter.register_from_python("not_equal", partial(make_int_bin_op, not_equal))
    rewriter.register_from_python("less", partial(make_int_bin_op, less))
    rewriter.register_from_python("less_equal", partial(make_int_bin_op, less_equal))
    rewriter.register_from_python("greater", partial(make_int_bin_op, greater))
    rewriter.register_from_python("greater_equal", partial(make_int_bin_op, greater_equal))
    rewriter.run(graph)
