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
from ..extensions.known_width import known_width
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

    def run(self, graph: Graph, is_convertible: Callable):
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
            if (
                node.operation == Operation.Generic
                and node.properties["name"] in self.patterns
            ):
                print("FOUND", [str(i) for i in node.inputs])
                print("FOUND", [i.is_scalar for i in node.inputs])
                assert len(node.inputs) == 2

                variable_input_indices = [
                    idx
                    for idx, pred in enumerate(graph.ordered_preds_of(node))
                    if not pred.operation == Operation.Constant
                ]
                print(variable_input_indices)
                if len(variable_input_indices) != 2:
                    continue
                sub_graph = self.find_sub_graph_for(
                    node.properties["name"],
                    node.inputs[0],
                    node.inputs[1],
                    node.output,
                )
                if is_convertible(sub_graph):
                    graph.expand_node(node, sub_graph)


def bitwise_and(bit_width, x, y):
    """
    table = LookupTable([0, 0, 1])
    z = 0
    for i in range(0):
        lsb_x = (x >> i) & 1
        lsb_y = (y >> i) & 1
        lsb = lsb_x + lsb_y
        z += table[lsb] << i
    return z
    """
    # x = 0
    # y = 0
    chunk_size = 2
    table_and = LookupTable(
        [x & y for x in range(1 << chunk_size) for y in range(1 << chunk_size)]
    )
    # print(table_and)
    z = known_width(0, bit_width)
    for offset in range(0, bit_width, chunk_size):
        # print(f"[{offset}]", z, bin(z))
        bit_width_this = min(chunk_size, bit_width - offset + 1)
        # print(offset, bit_width, bit_width_this)
        x_l = x & known_width(0b1111, bit_width_this)
        y_l = y & known_width(0b1111, bit_width_this)
        # print(bin(x), bin(x_l))
        # print(bin(y), bin(y_l))
        x = x >> chunk_size
        y = y >> chunk_size
        t = known_width((x_l << chunk_size) + y_l, 2 * chunk_size)
        # print(t, bin(t), table_and[t])
        # print("=>", known_width(table_and[t], chunk_size) << offset)
        z += known_width(known_width(table_and[t], chunk_size) << offset, bit_width)
    # print("@", z, bin(z))
    # assert False
    return z

    # x = 256
    # y = 256
    z = 0
    chunk_size = 8
    for offset in range(0, bit_width, chunk_size):
        # The +1 is important: it allows to have one bit of margin
        # for the possible carry
        bit_width_this = min(chunk_size, bit_width - offset + 1)
        # print(offset, bit_width, bit_width_this)
        x_1 = x & known_width(0b10101010, bit_width_this)
        x_2 = x & known_width(0b01010101, bit_width_this)
        y_1 = y & known_width(0b10101010, bit_width_this)
        y_2 = y & known_width(0b01010101, bit_width_this)
        # print(bin(x), bin(x_1), bin(x_2))
        # print(bin(y), bin(y_1), bin(y_2))
        x = x >> chunk_size
        y = y >> chunk_size
        z_1 = ((x_1 >> 1) + (y_1 >> 1)) & known_width(0b10101010, bit_width_this)
        z_2 = (x_2 + y_2) & known_width(0b10101010, bit_width_this)
        # print(bin(z_1), bin(z_2))
        z_ = z_1 + (z_2 >> 1)
        # print(bin(z_))
        z += z_ << offset
    # assert False
    return z
    """
    table_and = LookupTable([x & y for x in range(1 << 4) for y in range(1 << 4)])
    # print(table_and)
    z = 0
    for offset in range(0, bit_width, 4):
        x_g = x & 0b1111
        y_g = y & 0b1111
        x >>= 4
        y >>= 4
        # if x_g and y_g:
        #    print(bin(x_g), bin(y_g), bin(table[(x_g << 4) + y_g]))
        z += table_and[known_width((x_g * 16) + y_g, 8)] << offset
    return z
    """


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


def lookup_table_from_dict(d):
    lst = list(d.items())
    lst.sort()
    assert [a for a, _ in lst] == list(range(len(lst)))
    return LookupTable([b for _, b in lst])


def less_unsigned(x_type, y_type, x, y):
    chunk_size = 2
    table = LookupTable([0, 1, 1, 1])
    table_sign = lookup_table_from_dict(
        {0b000: 0, 0b001: 1, 0b100: 1, 0b101: 1, 0b010: 0, 0b011: 1, 0b110: 1, 0b111: 0}
    )

    out = 0
    x = (1 << 8) + x
    y = (1 << 8) + y
    # keep the last bit for the end
    print("-----------")
    print("###", x, bin(x), y, bin(y))
    for offset in range(0, bit_width - 1, chunk_size):
        bit_width_this = min(chunk_size, bit_width - 1 - offset)
        x_p = x & known_width(0b11111111, bit_width_this)
        y_p = y & known_width(0b11111111, bit_width_this)
        print(x_p, bin(x_p), y_p, bin(y_p), "|", bit_width_this, offset)
        x >>= bit_width_this
        y >>= bit_width_this
        c = x_p < y_p
        # Index by 2*c + out, which is a 2-bit width of the form c|out
        print("result:", table[(c << 1) + out])
        out = table[(c << 1) + out]
    # Now adjust the comparison depending on the sign:
    # Normally what remains in x or y are the bit representing the sign
    print("looking up: ", bin((x << 2) + (y << 1) + c))
    return table_sign[(x << 2) + (y << 1) + c]
    return out


def make_chunks(x_type, x, chunk_size=4):
    chunks = []
    this_chunk_size = chunk_size
    for offset in range(0, x_type.bit_width, chunk_size):
        this_chunk_size = min(chunk_size, x_type.bit_width - offset)
        chunks.append(x & known_width(0b11111111, this_chunk_size))
        x >>= this_chunk_size
    return chunks, this_chunk_size


C_UNX = {}
C_X = {}


def generate_table_signed(x_size, y_size, op):
    if (x_size, y_size) in C_X:
        return C_X[(x_size, y_size)]
    d = {}
    for x in range(-(1 << (x_size - 1)), 1 << (x_size - 1)):
        for y in range(-(1 << (y_size - 1)), 1 << (y_size - 1)):
            ty = y + (1 << y_size) if y < 0 else y
            tx = x + (1 << x_size) if x < 0 else x
            d[((tx) << y_size) | ty] = int(op(x, y))
    # print(len(d))
    C_X[(x_size, y_size)] = lookup_table_from_dict(d)
    return lookup_table_from_dict(d)


def generate_table_unsigned(size, op):
    if (size) in C_UNX:
        return C_UNX[(size, size)]
    d = {}
    for x in range(1 << size):
        for y in range(1 << size):
            d[(x << size) | y] = int(op(x, y))
    C_UNX[(size, size)] = lookup_table_from_dict(d)
    return lookup_table_from_dict(d)


def less(x_type, y_type, x, y):
    # return (((x >= 0 && y > 0) || (x < 0 && y <= 0)) && (x - y < 0))
    #        || (x < 0 && y > 0);
    chunk_size = 3
    chunks_x, x_last_chunk_width = make_chunks(x_type, x, chunk_size=chunk_size)
    chunks_y, y_last_chunk_width = make_chunks(y_type, y, chunk_size=chunk_size)
    diff_chunk_sizes = len(chunks_x) - len(chunks_y)
    if diff_chunk_sizes == 0:
        pass
    elif diff_chunk_sizes > 0:
        chunks_y += [0] * diff_chunk_sizes
    else:
        chunks_x += [0] * -diff_chunk_sizes

    out = 0

    table_unsigned = generate_table_unsigned(chunk_size, lambda x, y: x < y)
    table_signed = generate_table_signed(
        x_last_chunk_width, y_last_chunk_width, lambda x, y: x < y
    )
    table_combine = lookup_table_from_dict({0b01: 1, 0b00: 0, 0b11: 1, 0b10: 1})
    assert len(chunks_x) == len(chunks_y)
    print(bin(x), bin(y))
    out = 0
    for i, (chunk_x, chunk_y) in enumerate(zip(reversed(chunks_x), reversed(chunks_y))):
        if i == 0:
            out = table_signed[(chunk_x << y_last_chunk_width) + chunk_y]
            print(bin(chunk_x), bin(chunk_y), "=>", out)

        else:
            c = table_unsigned[(chunk_x << chunk_size) + chunk_y]
            lout = out
            out = table_combine[(out << 1) + c]
            print(bin(chunk_x), bin(chunk_y), "=>", c, lout, "@", out)
    return out
    sign_x = []
    sign_y = []
    chunk_size = 2
    return less_unsigned(x_type, y_type, x, y)

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
    print("TYPES:", x_type, y_type)
    assert_that(isinstance(x_type, Integer))
    assert_that(isinstance(y_type, Integer))
    return partial(callback, max(x_type.bit_width, y_type.bit_width))


def make_int_bin_op_bis(
    callback, x_type: BaseDataType, y_type: BaseDataType, output_type: BaseDataType
) -> Callable[[Any, Any], Any]:
    print("TYPES:", x_type, y_type)
    assert_that(isinstance(x_type, Integer))
    assert_that(isinstance(y_type, Integer))
    return partial(callback, x_type, y_type)


def rewrite_all_binops(graph, is_convertible: Callable):
    rewriter = BinaryNodeRewriter()
    rewriter.register_from_python("bitwise_or", partial(make_int_bin_op, bitwise_or))
    rewriter.register_from_python("bitwise_and", partial(make_int_bin_op, bitwise_and))
    rewriter.register_from_python("bitwise_xor", partial(make_int_bin_op, bitwise_xor))
    rewriter.register_from_python("equal", partial(make_int_bin_op, equal))
    rewriter.register_from_python("not_equal", partial(make_int_bin_op, not_equal))
    rewriter.register_from_python("less", partial(make_int_bin_op_bis, less))
    rewriter.register_from_python("less_equal", partial(make_int_bin_op, less_equal))
    rewriter.register_from_python("greater", partial(make_int_bin_op, greater))
    rewriter.register_from_python(
        "greater_equal", partial(make_int_bin_op, greater_equal)
    )
    rewriter.run(graph, is_convertible)
