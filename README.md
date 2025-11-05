# GPUPlayground
GPU hacking with Python and MLIR on HIP


## Install

### Pip

```sh
pip install -r requirements-hip.txt
pip install -r requirements.txt
pip install -r requirements-mlir.txt
```

### Local MLIR Build

```sh
git submodule update --init

pip install -r requirements-hip.txt
pip install -r requirements.txt
pip install -e .
```
