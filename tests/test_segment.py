from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from pdi_lab.operations import segment_image


def test_fixed_threshold_includes_equal_and_separates_sides():
    image = np.array([[119, 120, 121]], dtype=np.uint8)
    result = segment_image(image, "fixed", threshold_value=120)
    assert result.tolist() == [[0, 255, 255]]
    assert set(np.unique(result)) == {0, 255}


def test_otsu_and_adaptive_are_binary():
    image = np.tile(np.array([0, 0, 255, 255], dtype=np.uint8), (8, 8))
    for method in ("otsu", "adaptive"):
        result = segment_image(image, method)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})


def test_unknown_method_fails():
    with pytest.raises(ValueError):
        segment_image(np.zeros((2, 2), dtype=np.uint8), "unknown")


def test_invalid_adaptive_block_fails():
    with pytest.raises(ValueError):
        segment_image(np.zeros((5, 5), dtype=np.uint8), "adaptive", adaptive_block=4)


def test_hsv_requires_color_image():
    with pytest.raises(ValueError):
        segment_image(np.zeros((4, 4), dtype=np.uint8), "hsv")


def test_hsv_mask_is_binary_and_same_dimensions():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[:] = (0, 0, 255)  # BGR vermelho
    result = segment_image(image, "hsv", hsv_lower=(0, 100, 100), hsv_upper=(10, 255, 255))
    assert result.shape == image.shape[:2]
    assert set(np.unique(result)) == {255}


def test_invert():
    image = np.array([[0, 255]], dtype=np.uint8)
    result = segment_image(image, "fixed", threshold_value=120, invert=True)
    assert result.tolist() == [[255, 0]]


def test_cli_segment_writes_valid_mask_and_reports_pixels(tmp_path):
    from pdi_lab.__main__ import main

    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    cv2.imwrite(str(source), np.array([[0, 120, 121]], dtype=np.uint8))
    code = main([
        "--input", str(source), "--output", str(output),
        "--operation", "segment", "--method", "fixed", "--threshold", "120",
    ])
    assert code == 0
    mask = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1, 3)
    assert mask.ndim == 2
    assert set(np.unique(mask)) == {0, 255}


def test_cli_unknown_method_fails(tmp_path):
    from pdi_lab.__main__ import main

    source = tmp_path / "source.png"
    output = tmp_path / "mask.png"
    cv2.imwrite(str(source), np.zeros((2, 2), dtype=np.uint8))
    code = main([
        "--input", str(source), "--output", str(output),
        "--operation", "segment", "--method", "unknown",
    ])
    assert code != 0
