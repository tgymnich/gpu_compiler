import subprocess

from pathlib import Path
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

import mlir

mlir_path = Path(mlir.__path__[0])
mlir_translate = mlir_path / "bin" / "mlir-translate"
llc = mlir_path / "bin" / "llc"
llvm_as = mlir_path / "bin" / "llvm-as"

def compile_mlir_to_hsaco(mlir_module_str: str, chip_type="gfx950"):
    """Compiles MLIR module string to hsaco code."""
    with Context():
        module = Module.parse(mlir_module_str)
        module, gpu_module = apply_gpu_pipeline(module, chip_type)
        hsaco = generate_hsaco(str(gpu_module), chip_type)

    return hsaco


def compile_mlir_to_llvm(mlir_module_str: str, chip_type="gfx950"):
    """Compiles MLIR module string to hsaco code."""
    with Context():
        module = Module.parse(mlir_module_str)
        module, gpu_module = apply_gpu_pipeline(module, chip_type)
        llvm = generate_llvm(str(gpu_module), chip_type)

    return llvm


def compile_mlir_to_bitcode(mlir_module_str: str, chip_type="gfx950"):
    """Compiles MLIR module string to hsaco code."""
    with Context():
        module = Module.parse(mlir_module_str)
        module, gpu_module = apply_gpu_pipeline(module, chip_type)
        llvm_ir = generate_llvm(str(gpu_module), chip_type)
        llvm_bc = generate_bitcode(llvm_ir)

    return llvm_bc



def apply_gpu_pipeline(module, chip_type="gfx950"):
    """Applies the GPU compilation pipeline to the MLIR module."""
    pm = PassManager()
    # pm.enable_ir_printing(print_after_change=True)
    pm.add("canonicalize")
    pm.add(
        "one-shot-bufferize{ bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map }"
    )
    pm.add("canonicalize")
    pm.add("convert-linalg-to-affine-loops")
    pm.add("func.func(affine-loop-invariant-code-motion)")
    pm.add("func.func(convert-affine-for-to-gpu)")
    pm.add("gpu-kernel-outlining")
    pm.add("lower-affine")
    pm.add("gpu-decompose-memrefs")
    pm.add("expand-strided-metadata")
    pm.add("normalize-memrefs")
    # pm.add("convert-scf-to-cf")
    pm.add(
        "gpu.module(convert-gpu-to-rocdl{index-bitwidth=0 use-bare-ptr-memref-call-conv })")
    pm.add(f"rocdl-attach-target{{chip={chip_type} wave64 O=3}}")
    # pm.add("convert-amdgpu-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.add(
        "gpu-to-llvm { use-bare-pointers-for-host use-bare-pointers-for-kernels }")
    pm.run(module.operation)

    gpu_module = extract_gpu_module(module)

    return module, gpu_module


def extract_gpu_module(module: Module) -> Module:
    """Extracts the GPU module from a transformed MLIR module."""
    # Navigate the operation tree to find the GPU module
    # Structure: module -> region[0] -> block[0] -> operations[1] (GPU host-device code)
    # -> region[0] -> block[0] -> operations[0] (GPU module)
    try:
        main_func_op = module.operation.regions[0].blocks[0].operations[1]
        gpu_module_op = main_func_op.regions[0].blocks[0].operations[0]

        # Create a new module from the GPU module operation
        gpu_module = Module.parse(str(gpu_module_op))
        return gpu_module
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to extract GPU module: {e}") from e


def generate_llvm(gpu_module_str, chip_type="gfx950"):
    """Generates llvm from an MLIR GPU module string."""
    llvm_ir_result = subprocess.run(
        [mlir_translate, "--mlir-to-llvmir", "-"],
        input=gpu_module_str,
        capture_output=True,
        text=True,
    )

    if llvm_ir_result.returncode != 0:
        print("Error generating LLVM IR:")
        print(llvm_ir_result.stderr)
        return None

    return llvm_ir_result.stdout

def generate_bitcode(llvm_ir):
    """Generates llvm bitcode from an LLVM IR module string."""
    
    process = subprocess.Popen(
        [llvm_as, "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate(llvm_ir.encode("utf-8"))

    if process.returncode != 0:
        print("Error generating bitcode:")
        print(stderr)
        return None

    return stdout    

def generate_hsaco(gpu_module_str, chip_type="gfx950"):
    """Generates hsaco from an MLIR GPU module string."""
    llvm_ir = generate_llvm(gpu_module_str, chip_type)

    process = subprocess.Popen(
        [llc, "-mtriple=amdgcn", f"-mcpu={chip_type}", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate(llvm_ir.encode("utf-8"))

    if process.returncode != 0:
        print("Error generating hsaco:")
        print(stderr)
        return None

    return stdout.decode("utf-8")
