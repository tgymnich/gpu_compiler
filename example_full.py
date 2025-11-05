import ctypes
import numpy as np

from compile import compile_mlir_to_bitcode
from run import HipContext

# Example MLIR module for a matrix squaring operation
SQUARE_MLIR = """
module {
  func.func @square(%input: tensor<16x16xf32>, %output: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %x0 = linalg.square ins(%input : tensor<16x16xf32>) outs(%output : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %x0 : tensor<16x16xf32>
  }
}
"""

# Input data: 10x10 random matrix
size = 16
input_data = np.random.randn(size, size).astype(np.float32)

# Expected output for verification
expected_output = input_data * input_data

print("Compiling MLIR to HSACO...")
llvm_bc = compile_mlir_to_bitcode(SQUARE_MLIR)

if not llvm_bc:
    raise RuntimeError("HSACO compilation failed.")

with HipContext() as ctx:
    print("Running kernel on GPU...")

    # Create device arrays
    d_input = ctx.array(input_data)
    d_output = ctx.array(shape=(size, size), dtype=np.float32)

    # Execute kernel
    # The kernel_name should match the name generated during compilation
    ctx.run_kernel(
        llvm_bc,
        "square_kernel",
        [ctypes.c_int64(0), ctypes.c_int64(0), d_input, d_output],
        n=size * size,
        block_dims=(16, 1, 1),
    )

    # Get results back
    d_output.copy_device_to_host()
    result = d_output.host_array

    # Verify results
    print("Verifying results...")
    np.testing.assert_allclose(result, expected_output, rtol=1e-5)
    print("Success! Results verified.")
