import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, size, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, block_size) + pid * block_size
    mask = offset < size

    x = tl.load(x_ptr + offset, mask)
    y = tl.load(y_ptr + offset, mask)
    z = x + y

    tl.store(z_ptr + offset, z, mask)


def add(x, y):
    z = torch.empty_like(x, device="cuda")
    size = z.numel()

    def grid(meta):
        return (triton.cdiv(size, meta["block_size"]),)

    add_kernel[grid](x, y, z, size, 1024)

    return z


def main():
    size = 2**16
    print("size: ", size)

    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    print("x: ", x)
    print("y: ", y)

    r_tri = add(x, y)
    r_torch = x + y
    print("triton: ", r_tri)
    print("torch : ", r_torch)

    assert torch.allclose(r_tri, r_torch)


if __name__ == "__main__":
    main()