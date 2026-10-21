# [preannotation] - START
"""Where the pre-annotation runs its ONNX models, and with how many CPU threads.

The GPU when the installed onnxruntime can use it (onnxruntime-gpu, extra "preannotate-gpu"
in pyproject.toml), otherwise the CPU. NNSLR_PREANNOTATE_DEVICE=cpu forces the CPU.
NNSLR_PREANNOTATE_THREADS sets the threads of each model; helpers/1_prepare.sh sets it to
the cores divided by the routes prepared at the same time. Without it: half of the cores
(the panel prepares 2 routes at a time).
"""
from __future__ import annotations

import os

THREADS_ENV = 'NNSLR_PREANNOTATE_THREADS'
DEVICE_ENV = 'NNSLR_PREANNOTATE_DEVICE'
CUDA = 'CUDAExecutionProvider'
CPU = 'CPUExecutionProvider'


def threads() -> int:
    value = os.environ.get(THREADS_ENV, '')
    if value.isdigit() and int(value) > 0:
        return int(value)
    cores = os.cpu_count() or 4
    if cores // 2 < 2:
        return 2
    return cores // 2


def providers(ort) -> list[str]:
    """[CUDA, CPU] when onnxruntime-gpu is installed and not turned off, otherwise [CPU]."""
    if os.environ.get(DEVICE_ENV, '') == 'cpu':
        return [CPU]
    if CUDA not in ort.get_available_providers():
        return [CPU]
    # onnxruntime-gpu[cuda,cudnn] brings the CUDA and cuDNN libraries as nvidia-* wheels:
    # they are not on the library path, preload_dlls loads them from site-packages.
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    return [CUDA, CPU]


def session_options(ort):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads()
    options.inter_op_num_threads = 1
    return options
# [preannotation] - END
