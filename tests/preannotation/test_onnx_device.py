"""GPU or CPU for the pre-annotation models, and their threads, with a fake onnxruntime."""
from __future__ import annotations

import sys
import types

import pytest

from nnslr_tools import onnx_device


class FakeSession:
    def __init__(self, providers, broken_kernels=False):
        self.providers = providers
        self.broken_kernels = broken_kernels

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return [type('Input', (), {'name': 'images'})()]

    def run(self, outputs, feed):
        if self.broken_kernels:
            raise RuntimeError('Non-zero status code returned while running Conv node.\nno kernel image is available')
        return [feed['images'].shape]


class FakeOrt:
    """onnxruntime with or without CUDA; broken_gpu makes the CUDA session fail as on the V100 with CUDA 13."""

    def __init__(self, cuda: bool, broken_gpu: bool = False, broken_kernels: bool = False):
        self.cuda = cuda
        self.broken_gpu = broken_gpu
        self.broken_kernels = broken_kernels
        self.preloaded = False
        self.sessions = []

    def get_available_providers(self):
        if self.cuda:
            return [onnx_device.CUDA, onnx_device.CPU]
        return [onnx_device.CPU]

    def preload_dlls(self):
        self.preloaded = True

    def SessionOptions(self):
        return type('Options', (), {})()

    def InferenceSession(self, path, sess_options, providers):
        self.sessions.append((providers, sess_options.intra_op_num_threads))
        if onnx_device.CUDA in providers and self.broken_gpu:
            raise RuntimeError('Exception during initialization:\nCUBLAS failure 8: architectural feature absent')
        return FakeSession(providers, broken_kernels=self.broken_kernels and onnx_device.CUDA in providers)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(onnx_device, '_gpu_failed', False)
    monkeypatch.delenv(onnx_device.DEVICE_ENV, raising=False)
    monkeypatch.delenv(onnx_device.THREADS_ENV, raising=False)


def test_gpu_when_onnxruntime_gpu_works():
    ort = FakeOrt(cuda=True)
    session = onnx_device.create_session(ort, 'model.onnx')
    assert onnx_device.on_gpu(session) and ort.preloaded


def test_cpu_with_the_cpu_onnxruntime_or_when_forced(monkeypatch):
    assert not onnx_device.on_gpu(onnx_device.create_session(FakeOrt(cuda=False), 'model.onnx'))
    monkeypatch.setenv(onnx_device.DEVICE_ENV, 'cpu')
    ort = FakeOrt(cuda=True)
    assert not onnx_device.on_gpu(onnx_device.create_session(ort, 'model.onnx'))
    assert ort.preloaded is False


def test_a_gpu_that_fails_falls_back_to_the_cpu_once(capsys):
    ort = FakeOrt(cuda=True, broken_gpu=True)
    first = onnx_device.create_session(ort, 'detector.onnx')
    second = onnx_device.create_session(ort, 'surface.onnx')
    assert not onnx_device.on_gpu(first) and not onnx_device.on_gpu(second)
    # The GPU is tried only once per process.
    assert [providers for providers, _ in ort.sessions] == [
        [onnx_device.CUDA, onnx_device.CPU], [onnx_device.CPU], [onnx_device.CPU]]
    assert 'GPU not usable, using the CPU: CUBLAS failure 8' in capsys.readouterr().err


class FakeArray:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture
def fake_numpy(monkeypatch):
    # The probe builds its zeros with numpy, which the core test environment does not have.
    module = types.ModuleType('numpy')
    module.float32 = 'float32'
    module.zeros = lambda shape, dtype: FakeArray(shape)
    monkeypatch.setitem(sys.modules, 'numpy', module)


def test_the_probe_keeps_a_working_gpu(fake_numpy):
    detector = onnx_device.create_session(FakeOrt(cuda=True), 'detector.onnx', probe_shape=(1, 3, 640, 640))
    assert onnx_device.on_gpu(detector)


def test_a_gpu_without_kernels_fails_the_probe_and_falls_back(fake_numpy, capsys):
    ort = FakeOrt(cuda=True, broken_kernels=True)
    detector = onnx_device.create_session(ort, 'detector.onnx', probe_shape=(1, 3, 640, 640))
    assert not onnx_device.on_gpu(detector)
    assert 'GPU not usable, using the CPU: no kernel image is available' in capsys.readouterr().err
    # The road models and the OCR of the same worker go straight to the CPU.
    assert not onnx_device.on_gpu(onnx_device.create_session(ort, 'surface.onnx'))


def test_threads_from_the_environment_or_half_of_the_cores(monkeypatch):
    monkeypatch.setattr(onnx_device.os, 'cpu_count', lambda: 20)
    assert onnx_device.threads() == 10
    monkeypatch.setattr(onnx_device.os, 'cpu_count', lambda: 2)
    assert onnx_device.threads() == 2
    monkeypatch.setenv(onnx_device.THREADS_ENV, '7')
    ort = FakeOrt(cuda=False)
    onnx_device.create_session(ort, 'model.onnx')
    assert ort.sessions == [([onnx_device.CPU], 7)]
    monkeypatch.setenv(onnx_device.THREADS_ENV, 'many')
    assert onnx_device.threads() == 2
