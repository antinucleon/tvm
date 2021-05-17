
import tvm
from tvm.contrib import cblas

def gemm(a, b, trans_flag):
    transa = False
    transb = False
    if trans_flag == "NT":
        transb = True
    elif trans_flag == "TN":
        transa = True
    elif trans_flag == "TT":
        transa = True
        transb = True
    return cblas.matmul(a, b, transa, transb)