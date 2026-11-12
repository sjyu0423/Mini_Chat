import pytest
import torch

from mini_chatgpt.device import (
    DeviceUnavailableError,
    describe_device,
    resolve_dtype,
    select_device,
)


def probes(mps, cuda):
    return {"is_mps_available": lambda: mps, "is_cuda_available": lambda: cuda}


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [(True, False, "mps"), (False, False, "cpu"), (False, True, "cuda"), (True, True, "cuda")],
)
def test_auto_selection(mps, cuda, expected):
    assert select_device("auto", **probes(mps, cuda)).type == expected


def test_cpu_always_available():
    assert select_device("cpu", **probes(False, False)).type == "cpu"


def test_explicit_mps_when_unavailable_raises():
    with pytest.raises(DeviceUnavailableError):
        select_device("mps", **probes(False, False))


def test_explicit_cuda_when_unavailable_raises():
    with pytest.raises(DeviceUnavailableError):
        select_device("cuda", **probes(True, False))


def test_explicit_mps_when_available():
    assert select_device("MPS", **probes(True, False)).type == "mps"


def test_unknown_preference():
    with pytest.raises(ValueError):
        select_device("tpu")


def test_dtype_resolution():
    cpu = torch.device("cpu")
    assert resolve_dtype("float32", cpu) is torch.float32
    assert resolve_dtype("bfloat16", cpu) is torch.bfloat16
    with pytest.raises(ValueError):
        resolve_dtype("float16", cpu)
    with pytest.raises(ValueError):
        resolve_dtype("int8", cpu)


def test_real_auto_device_works():
    dev = select_device("auto")
    x = torch.arange(4.0, device=dev)
    assert float((x * 2).sum().cpu()) == 12.0
    assert describe_device(dev).selected == str(dev)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_selected_on_apple_silicon():
    assert select_device("auto").type in {"mps", "cuda"}
