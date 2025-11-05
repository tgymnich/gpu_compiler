from compile import compile_mlir_to_hsaco, compile_mlir_to_llvm

SQUARE_MLIR = """
module {
  func.func @square(%input: tensor<10x10xf32>, %output: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %x0 = linalg.square ins(%input : tensor<10x10xf32>) outs(%output : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %x0 : tensor<10x10xf32>
  }
}
"""

def main():
    hsaco_code = compile_mlir_to_hsaco(SQUARE_MLIR)
    print(hsaco_code)
    
    llvm_ir = compile_mlir_to_llvm(SQUARE_MLIR)
    print(llvm_ir)

if __name__ == "__main__":
    main()
