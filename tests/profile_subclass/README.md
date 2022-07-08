# Profiling the performance of subclassing/`__torch_function__` in MONAI

## Requirements
```bash
pip install py-spy
pip install snakeviz  # for viewing the cProfile results
```

## Commands

### Install MONAI
```
./runtests.sh --build   # from monai's root directory
```
or follow the installation guide (https://docs.monai.io/en/latest/installation.html)

### Profiling the task of adding two MetaTensors
```bash
python profiling.py
```

### Profiling using `py-spy`
```bash
py-spy record -o Tensor.svg -- python pyspy_profiling.py Tensor
py-spy record -o SubTensor.svg -- python pyspy_profiling.py SubTensor
py-spy record -o SubWithTorchFunc.svg -- python pyspy_profiling.py SubWithTorchFunc
py-spy record -o MetaTensor.svg -- python pyspy_profiling.py MetaTensor
```

### Profiling using `cProfile` and `SNAKEVIZ`

```bash
python cprofile_profiling.py
snakeviz out_200.prof
```

---
These tests are based on the following code:
https://github.com/pytorch/pytorch/tree/v1.11.0/benchmarks/overrides_benchmark

- Overhead for torch functions when run on `torch.Tensor` objects is on the order of 2 microseconds.
- `__torch_function__` should add zero overhead for `torch.Tensor` inputs, a small overhead for subclasses of `torch.Tensor`, and an order of microseconds for `MeatTensor`.
- Changing the dispatching mechanism may result in changes that are on the order of 100 ns, which are hard to detect due to noise, but important.
