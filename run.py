from hip import hip, hiprtc

import numpy as np
import ctypes


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        msg = hiprtc.hiprtcGetErrorString(err)
        raise RuntimeError(f"{str(err)}: {msg}")
    return result


def setup_hip(device_id=None):
    """Initialize HIP and create a context."""
    print("Initializing HIP...")

    hip_check(hip.hipInit(0))

    if device_id is None:
        device_id = 0

    device = hip_check(hip.hipDeviceGet(device_id))
    context = hip_check(hip.hipCtxCreate(0, device))

    print(f"HIP context created on device {device_id}.")
    return context


def cleanup_hip(context):
    """Destroy the HIP context."""
    if context:
        print("Destroying HIP context...")
        hip_check(hip.hipCtxDestroy(context))
        print("HIP context destroyed.")


class HipArray:
    """Class to manage GPU memory and transfers with automatic cleanup."""

    def __init__(self, host_array=None, shape=None, dtype=np.float32):
        """Initialize a HIP array either from host data or empty with given shape."""
        self.device_ptr = None
        self.nbytes = 0
        self.dtype = dtype
        self.shape = None

        # Create from host array
        if host_array is not None:
            if not isinstance(host_array, np.ndarray):
                host_array = np.array(host_array, dtype=dtype)
            self.host_array = host_array
            self.shape = host_array.shape
            self.nbytes = host_array.nbytes
            self.dtype = host_array.dtype
            self.device_ptr = hip_check(hip.hipMalloc(self.nbytes))
            self.copy_host_to_device()

        # Create empty with shape
        elif shape is not None:
            self.host_array = np.zeros(shape, dtype=dtype)
            self.shape = shape
            self.nbytes = self.host_array.nbytes
            self.device_ptr = hip_check(hip.hipMalloc(self.nbytes))

        else:
            raise ValueError("Either host_array or shape must be provided")

    def copy_host_to_device(self):
        """Copy data from host to device."""
        if not self.host_array.flags.c_contiguous:
            self.host_array = np.ascontiguousarray(self.host_array)
        hip_check(
            hip.hipMemcpy(self.device_ptr, self.host_array.ctypes.data,
                          self.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice)
        )

    def copy_device_to_host(self):
        """Copy data from device to host."""
        hip_check(
            hip.hipMemcpy(self.host_array.ctypes.data, self.device_ptr,
                          self.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
        )

    def free(self):
        """Free GPU memory."""
        if self.device_ptr:
            hip_check(hip.hipFree(self.device_ptr))
            self.device_ptr = None

    def __del__(self):
        """Automatically free GPU memory when object is deleted."""
        self.free()


class HipContext:
    """Context manager for HIP operations to ensure proper cleanup."""

    def __init__(self, device_id=0):
        self.context = None
        self.device_id = device_id
        self.arrays = []

    def __enter__(self):
        self.context = setup_hip(self.device_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up any arrays that were created
        for array in self.arrays:
            array.free()

        # Destroy the HIP context
        if self.context:
            cleanup_hip(self.context)

    def array(self, host_array=None, shape=None, dtype=np.float32):
        """Create a HipArray and register it for automatic cleanup."""
        array = HipArray(host_array, shape, dtype)
        self.arrays.append(array)
        return array

    def run_kernel(
        self,
        llvm_bc,
        kernel_name,
        args,
        n=None,
        grid_dims=None,
        block_dims=(128, 1, 1),
    ):
        """Run a kernel with automatic dimension calculation if needed."""
        # Prepare arguments
        kernel_args = []
        
        for arg in args:
            if isinstance(arg, HipArray):
                kernel_args.append(arg.device_ptr)
            elif isinstance(arg, int):
                kernel_params.append(ctypes.c_int(arg))
            elif isinstance(arg, float):
                kernel_params.append(ctypes.c_float(arg))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Prepare grid and block dimensions
        if grid_dims is not None:
            # Use explicit grid dimensions
            grid = list(grid_dims) + [1] * (3 - len(grid_dims))
            block = list(block_dims) + [1] * (3 - len(block_dims))
        elif n is not None:
            # Calculate grid dimensions from n
            threads_per_block = (
                block_dims[0] if isinstance(block_dims, tuple) else block_dims
            )
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            grid = (blocks_per_grid, 1, 1)
            block = (threads_per_block, 1, 1)
        else:
            raise ValueError("Either n or grid_dims must be provided")


        # link_state = hip_check(hiprtc.hiprtcLinkCreate(0, None, None))
        link_state = hip_check(hiprtc.ext.hiprtcLinkCreate2(
                HIPRTC_JIT_GENERATE_DEBUG_INFO=1,
                HIPRTC_JIT_GENERATE_LINE_INFO=1,
            )
        )
        hip_check(hiprtc.hiprtcLinkAddData(link_state,
                hiprtc.hiprtcJITInputType.hipJitInputLLVMBitcode,
                llvm_bc,
                len(llvm_bc),
                kernel_name.encode("utf-8"),
                0,
                None,
                None
            )
        )

        hsaco_object, code_size = hip_check(hiprtc.hiprtcLinkComplete(link_state))

        module = hip_check(hip.hipModuleLoadData(hsaco_object))

        kernel_func = hip_check(
            hip.hipModuleGetFunction(module, kernel_name.encode("utf-8"))
        )

        hip_check(
            hip.hipModuleLaunchKernel(
                kernel_func,
                *grid,
                *block,
                sharedMemBytes=0,
                stream=None,
                kernelParams=None,
                extra=kernel_args,
            )
        )

        hip_check(hip.hipDeviceSynchronize())
        hip_check(hip.hipModuleUnload(module))
        hip_check(hiprtc.hiprtcLinkDestroy(link_state))
