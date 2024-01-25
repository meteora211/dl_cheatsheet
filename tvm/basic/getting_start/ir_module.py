from __future__ import annotations
import tvm
import numpy as np
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm import relax

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def relu0(X: T.Buffer((1, 128), "float32"),
              Y: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "relu0", "tir.noalias": True})
        for i, j in T.grid(1, 128):
            with T.block("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @T.prim_func
    def linear0(X: T.Buffer((1, 784), "float32"),
                W: T.Buffer((128, 784), "float32"),
                B: T.Buffer((128,), "float32"),
                Z: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
        Y = T.alloc_buffer((1, 128), "float32")
        for i, j, k in T.grid(1, 128, 784):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]

        for i, j in T.grid(1, 128):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = B[vj] + Y[vi, vj]

    @T.prim_func
    def linear1(X: T.Buffer((1, 128), "float32"),
                W: T.Buffer((10, 128), "float32"),
                B: T.Buffer((10,), "float32"),
                Z: T.Buffer((1, 10), "float32")):
        T.func_attr({"global_symbol": "linear1", "tir.noalias": True})
        Y = T.alloc_buffer((1, 10), "float32")
        for i, j, k in T.grid(1, 10, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]

        for i, j in T.grid(1, 10):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = B[vj] + Y[vi, vj]

    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w0: R.Tensor((128, 784), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((10, 128), "float32"),
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            cls = MyModule
            lv0 = R.call_tir(cls.linear0, (x, w0, b0), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_tir(cls.relu0, (lv0, ), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            out = R.call_tir(cls.linear1, (lv1, w1, b1), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out

MyModule.show()

ex = relax.build(MyModule, target="llvm")
print(type(ex))
vm = relax.VirtualMachine(ex, tvm.cpu())
data_nd = tvm.nd.array(np.random.rand(1, 784).astype("float32"))
w0_nd = tvm.nd.array(np.random.rand(128, 784).astype("float32"))
b0_nd = tvm.nd.array(np.random.rand(128,).astype("float32"))
w1_nd = tvm.nd.array(np.random.rand(10, 128).astype("float32"))
b1_nd = tvm.nd.array(np.random.rand(10,).astype("float32"))

nd_res = vm["main"](data_nd, w0_nd, b0_nd, w1_nd, b1_nd)
print(nd_res)
