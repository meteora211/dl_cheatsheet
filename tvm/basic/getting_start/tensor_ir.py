import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# define tensorIR
@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer((4, 4), "int64"),
            B: T.Buffer((4, 4), "int64"),
            C: T.Buffer((4, 4), "int64"),):
        T.func_attr({"global_symbol": "add"})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]

print(type(MyAdd))
print(type(MyAdd["add"]))

# numpy array
a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)
c_np = a + b

# build target
rt_lib = tvm.build(MyAdd, target="llvm")

# input
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))

# execute
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)

# a matmul-relu example
@tvm.script.ir_module
class MyMatmul:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
            B: T.Buffer((128, 128), "float32"),
            C: T.Buffer((128, 128), "float32"),):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([128, 128], dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

# show code
MyMatmul.show()

# transform
sch = tvm.tir.Schedule(MyMatmul)
block_Y = sch.get_block("Y", func_name = "mm_relu")
i, j, k = sch.get_loops(block_Y)
j0, j1 = sch.split(j, factors=[None, 4])
# check the for-loop is splited
sch.mod.show()

sch.reorder(j0, k, j1)
block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)
sch.mod.show()


block_Y = sch.get_block("Y", "mm_relu")
sch.decompose_reduction(block_Y, k)
sch.mod.show()

print(sch.trace)

# build target
rt_lib = tvm.build(MyMatmul, target="llvm")

# input
a_tvm = tvm.nd.array(np.random.rand(128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.empty((128, 128), dtype=np.float32))

# execute
rt_lib["mm_relu"](a_tvm, b_tvm, c_tvm)
