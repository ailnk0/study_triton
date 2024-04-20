import triton
import triton.language as tl

@triton.jit
def hello_triton():
    tl.device_print("hello world!")

hello_triton[(1,)]()
