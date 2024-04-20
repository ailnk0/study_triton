import triton
import triton.language as tl

@triton.jit
def print_grid():
    x_pid = tl.program_id(0)
    y_pid = tl.program_id(1)
    z_pid = tl.program_id(2)
    tl.device_print("x_pid:", x_pid)
    tl.device_print("y_pid:", y_pid)
    tl.device_print("z_pid:", z_pid)

def grid(meta):
    return (1, 2, 4)

print_grid[grid]()
