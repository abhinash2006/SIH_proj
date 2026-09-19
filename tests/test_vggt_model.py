import os
import numpy as np
import pytest
from src.geometry.vggt_model import (
    load_vggt,
    MockVGGTAdapter,
    PretrainedVGGTAdapter,
    FineTunedVGGTAdapter,
    VGGTOutput,
)


def test_mock_vggt_adapter(tmp_path):
    # Test mock adapter predictions
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img1.write_bytes(b"dummy1")
    img2.write_bytes(b"dummy2")

    adapter = MockVGGTAdapter(resolution=64)
    adapter.load()
    output = adapter.predict([str(img1), str(img2)])

    assert isinstance(output, VGGTOutput)
    assert output.num_frames == 2
    assert output.extrinsics.shape == (2, 3, 4)
    assert output.intrinsics.shape == (2, 3, 3)
    assert output.depth_maps.shape == (2, 64, 64, 1)
    assert output.point_maps.shape == (2, 64, 64, 3)
    assert output.images.shape == (2, 64, 64, 3)
    assert output.backend_type == "mock"

    adapter.unload()


def test_vggt_factory():
    # Test factory switching
    mock_engine = load_vggt(checkpoint_type="mock")
    assert isinstance(mock_engine, MockVGGTAdapter)

    pre_engine = load_vggt(checkpoint_type="pretrained", checkpoint_path="models/VGGT-1B/model.pt")
    assert isinstance(pre_engine, PretrainedVGGTAdapter)

    ft_engine = load_vggt(checkpoint_type="finetuned", checkpoint_path="models/vggt/finetuned/test.pth")
    assert isinstance(ft_engine, FineTunedVGGTAdapter)


def test_vggt_env_override(monkeypatch):
    monkeypatch.setenv("VGGT_CHECKPOINT", "custom_weights.pth")
    engine = load_vggt(checkpoint_type="pretrained")
    assert engine.checkpoint_path == "custom_weights.pth"
