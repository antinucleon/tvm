import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor as runtime

M = 128
N = 32
K = 64


A = relay.var("a", shape=(M, K))
B = relay.var("b", shape=(K, N))
C = relay.nn.gemm(A, B, trans_flag="NN")
F = relay.Function([A, B], C)
from tvm.relay.testing import run_opt_pass, run_infer_type

params = {
  "a": np.random.uniform(-1, 1, (M, K)).astype("float32"),
  "b": np.random.uniform(-1, 1, (K, N)).astype("float32")
}

target = "cuda -libs=cublas"  # use cudnn for convolution
with tvm.transform.PassContext(opt_level=3):
  lib = relay.build_module.build(F, target=target, target_host="llvm", params={"a": params["a"]})

dev = tvm.device(target, 0)
module = runtime.GraphModule(lib["default"](dev))
#module.set_input("a", tvm.nd.array(params["a"]))
module.set_input("b", tvm.nd.array(params["b"]))
module.run()
out_shape = (M, N)
out = module.get_output(0, tvm.nd.empty(out_shape))
out_np = out.asnumpy()

np.testing.assert_allclose(out_np, np.dot(params["a"], params["b"]), atol=1e-2, rtol=1e-2)