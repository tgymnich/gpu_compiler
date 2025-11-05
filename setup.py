from setuptools import setup, Extension, find_packages
import subprocess
import os
from setuptools.command.build_ext import build_ext
from pathlib import Path
import sys


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if ext.name == "runtime":
                self.build_runtime(ext)
            elif ext.name == "mlir":
                self.build_mlir(ext)
            else:
                raise NotImplementedError("unknown cmake project")

    def build_mlir(self, ext):
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        install_dir = extdir / ext.name

        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        source_dir = Path(ext.sourcedir)
        llvm_dir = source_dir / "llvm-project"
        
        cmake_args = [
            f"-B{build_dir}",
            f"-G Ninja",
            "-DLLVM_TARGETS_TO_BUILD=host;AMDGPU",
            "-DLLVM_ENABLE_PROJECTS=mlir",
            "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DLLVM_ENABLE_ZSTD=OFF",
            "-DLLVM_INSTALL_UTILS=ON",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}"
        ]

        subprocess.check_call(["cmake", llvm_dir / "llvm", *cmake_args], cwd=build_dir)
        subprocess.check_call(["cmake",  "--build", ".", "--target", "install"], cwd=build_dir)


setup(
    name="gpu_playground",
    version="0.1.0",
    packages=find_packages(),
    package_dir={
        'mlir': 'mlir/python_packages/mlir_core/mlir'
    },
    include_package_data=True,
    python_requires=">=3.10",
    ext_modules=[
        CMakeExtension("mlir")
    ],
    install_requires=["hip-python, numpy"]
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
