"""Check the local CrossSim environment without downloading model datasets.

Run: uv run --locked python verify_environment.py
CPU and GPU checks use separate processes because CrossSim fixes its backend
after the first AnalogCore is constructed.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def check_core(gpu=False):
    import numpy as np
    from simulator import AnalogCore
    from applications.mvm_params import set_params

    rng = np.random.default_rng(42)
    weights = rng.uniform(-1, 1, (16, 24)).astype(np.float32)
    inputs = rng.uniform(-1, 1, 24).astype(np.float32)
    params = set_params(useGPU=gpu)
    if gpu:
        import cupy as cp

        print(f"CuPy {cp.__version__}; GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}", flush=True)
        actual = cp.asnumpy(AnalogCore(cp.asarray(weights), params=params) @ cp.asarray(inputs))
        cp.cuda.Stream.null.synchronize()
    else:
        actual = AnalogCore(weights, params=params) @ inputs
    np.testing.assert_allclose(actual, weights @ inputs, rtol=1e-4, atol=1e-5)
    print(f"PASS CrossSim {'GPU' if gpu else 'CPU'} MVM", flush=True)


def check_torch(gpu=False):
    import torch
    import torchvision
    from applications.mvm_params import set_params
    from simulator.algorithms.dnn.torch import from_torch

    if gpu:
        assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"
    device = "cuda" if gpu else "cpu"
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(24, 16)).to(device)
    analog = from_torch(model, set_params(useGPU=gpu))
    inputs = torch.randn(4, 24, device=device, requires_grad=True)
    actual = analog(inputs)
    torch.testing.assert_close(actual, model(inputs), rtol=1e-4, atol=1e-5)
    actual.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert analog[0].weight.grad is not None
    if gpu:
        torch.cuda.synchronize()
        print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA: {torch.version.cuda}", flush=True)
    print(f"PASS PyTorch {torch.__version__}, torchvision {torchvision.__version__}, CrossSim {device} forward/backward", flush=True)


def check_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import numpy as np
    import tensorflow as tf
    import keras
    from applications.mvm_params import set_params
    from simulator.algorithms.dnn.keras import AnalogDense

    inputs = np.random.default_rng(42).uniform(-1, 1, (4, 24)).astype(np.float32)
    dense = keras.layers.Dense(16)
    expected = dense(inputs).numpy()
    analog = AnalogDense(params=set_params(), units=16)
    analog(inputs)
    analog.set_weights(dense.get_weights())
    np.testing.assert_allclose(np.asarray(analog(inputs)), expected, rtol=1e-4, atol=1e-5)
    print(f"PASS TensorFlow {tf.__version__}, Keras {keras.__version__}, CrossSim AnalogDense CPU", flush=True)
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')} (expected [] on native Windows)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["core-cpu", "core-gpu", "torch-cpu", "torch-gpu", "tensorflow"])
    args = parser.parse_args()
    if args.case is None:
        print(f"Python: {sys.version.split()[0]}\nInterpreter: {sys.executable}", flush=True)
        for case in ["core-cpu", "core-gpu", "torch-cpu", "torch-gpu", "tensorflow"]:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--case", case], check=True)
        print("All environment checks passed.", flush=True)
    elif args.case.startswith("core-"):
        check_core(gpu=args.case.endswith("gpu"))
    elif args.case.startswith("torch-"):
        check_torch(gpu=args.case.endswith("gpu"))
    else:
        check_tensorflow()


if __name__ == "__main__":
    main()
