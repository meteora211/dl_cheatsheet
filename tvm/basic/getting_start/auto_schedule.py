import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm import meta_schedule as ms

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_factors = sch.sample_perfect_tile(loop=j, n=2)
    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch

def random_search(mod: tvm.IRModule, num_trials=5):
    best_result = None
    best_sch = None

    for i in range(num_trials):
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))
        lib = tvm.build(sch.mod, target = "llvm")
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_before(a_nd, b_nd, c_nd).mean

        print("=============Attempt %d, time-cost: %.3f ms=============" % (i, result * 1000))
        print(sch.trace)

        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch

    return best_sch

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm = a_np @ b_np

a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")

lib = tvm.build(MyModule, target="llvm")
f_timer_before = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule: %.3f ms" % (f_timer_before(a_nd, b_nd, c_nd).mean * 1000))


# stochastic schedule
sch = tvm.tir.Schedule(MyModule)
stochastic_schedule_mm(sch)
print("======================================================")
sch.mod.show()
print(sch.trace)
print("======================================================")



sch = random_search(MyModule)
print(sch.trace)


database = ms.tune_tir(
        mod = MyModule,
        target="llvm --num-cores=1",
        max_trials_global=64,
        num_trials_per_iter=64,
        # space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
        work_dir="./tune_tmp",
        # task_name="main",
        )
sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
lib = tvm.build(sch.mod, target="llvm")
f_timer_after = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule after tuning: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))

