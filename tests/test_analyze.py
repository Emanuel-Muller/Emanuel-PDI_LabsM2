import csv
import json

import cv2
import numpy as np
import pytest

from pdi_lab.analyze import analyze_mask
from pdi_lab.__main__ import main


def write_mask(path, array):
    assert cv2.imwrite(str(path), array)
    return str(path)


def test_diagonal_pixels_have_4_vs_8_components(tmp_path):
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 255
    mask[2, 2] = 255
    path = write_mask(tmp_path / "diagonal.png", mask)
    summary = analyze_mask(path, str(tmp_path / "diag"), connectivity=8,
                           kernel_size=1, open_iterations=0, close_iterations=1)
    assert summary["initial_connectivity_counts"] == {"4": 2, "8": 1}
    assert summary["initial"]["raw_foreground_area"] == 2


def test_opening_removes_isolated_noise_and_preserves_main_region(tmp_path):
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[4:11, 4:11] = 255
    mask[1, 1] = 255
    path = write_mask(tmp_path / "noise.png", mask)
    summary = analyze_mask(path, str(tmp_path / "noise_result"), connectivity=8,
                           kernel_size=3, open_iterations=1, close_iterations=0)
    final = cv2.imread(str(tmp_path / "noise_result_final.png"), cv2.IMREAD_UNCHANGED)
    assert final[1, 1] == 0
    assert np.count_nonzero(final) > 0
    assert summary["final"]["foreground_pixels"] < summary["initial"]["foreground_pixels"]


def test_closing_fills_small_hole(tmp_path):
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[3:12, 3:12] = 255
    mask[7, 7] = 0
    path = write_mask(tmp_path / "hole.png", mask)
    analyze_mask(path, str(tmp_path / "hole_result"), connectivity=8,
                 kernel_size=3, open_iterations=0, close_iterations=1)
    final = cv2.imread(str(tmp_path / "hole_result_final.png"), cv2.IMREAD_UNCHANGED)
    assert final[7, 7] == 255


def test_empty_mask_zero_components(tmp_path):
    path = write_mask(tmp_path / "empty.png", np.zeros((4, 6), dtype=np.uint8))
    summary = analyze_mask(path, str(tmp_path / "empty_result"))
    assert summary["initial"]["raw_count"] == 0
    assert summary["final"]["raw_count"] == 0
    assert summary["initial"]["area_sum_matches_foreground_pixels"]


def test_object_touching_border_is_counted(tmp_path):
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0:2, 0:3] = 255
    path = write_mask(tmp_path / "border.png", mask)
    summary = analyze_mask(path, str(tmp_path / "border_result"), kernel_size=1,
                           open_iterations=0, close_iterations=1)
    assert summary["initial"]["raw_count"] == 1
    with open(tmp_path / "border_result_initial.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert (int(row["x"]), int(row["y"])) == (0, 0)


def test_invalid_connectivity_cli_returns_nonzero(tmp_path):
    source = tmp_path / "mask.png"
    cv2.imwrite(str(source), np.zeros((5, 5), dtype=np.uint8))
    code = main(["--input", str(source), "--output", str(tmp_path / "out"),
                 "--operation", "analyze", "--connectivity", "6"])
    assert code != 0


def test_analyze_requires_binary_single_channel(tmp_path):
    path = tmp_path / "not_binary.png"
    cv2.imwrite(str(path), np.array([[0, 120], [255, 0]], dtype=np.uint8))
    with pytest.raises(ValueError, match="somente 0 e 255"):
        analyze_mask(str(path), str(tmp_path / "bad"))


def test_csv_area_sum_and_binary_stages(tmp_path):
    mask = np.zeros((12, 13), dtype=np.uint8)
    mask[2:7, 3:8] = 255
    mask[9, 11] = 255
    path = write_mask(tmp_path / "source.png", mask)
    summary = analyze_mask(path, str(tmp_path / "case"), min_area=2, kernel_size=1,
                           open_iterations=0, close_iterations=1)
    assert summary["initial"]["area_sum_matches_foreground_pixels"]
    assert summary["final"]["area_sum_matches_foreground_pixels"]
    for name in ("case_initial.png", "case_final.png"):
        result = cv2.imread(str(tmp_path / name), cv2.IMREAD_UNCHANGED)
        assert result.shape == mask.shape
        assert result.ndim == 2
        assert set(np.unique(result)).issubset({0, 255})
    saved = json.loads((tmp_path / "case_summary.json").read_text(encoding="utf-8"))
    assert saved["min_area"] == 2

@pytest.mark.parametrize("image_name", ["m1_shapes_border_256.png", "m1_color_scene_256.png"])
def test_m2_study_images_can_be_segmented_and_analyzed(tmp_path, image_name):
    from pathlib import Path
    from pdi_lab.operations import segment_image

    project_root = Path(__file__).resolve().parents[1]
    source_path = project_root / "images" / "input" / image_name
    source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    assert source is not None
    mask = segment_image(source, "fixed", threshold_value=120)
    mask_path = tmp_path / f"{image_name}_mask.png"
    assert cv2.imwrite(str(mask_path), mask)
    summary = analyze_mask(str(mask_path), str(tmp_path / image_name), connectivity=8,
                           min_area=20, kernel_shape="ellipse", kernel_size=3,
                           open_iterations=1, close_iterations=1)
    assert summary["initial"]["area_sum_matches_foreground_pixels"]
    assert summary["final"]["area_sum_matches_foreground_pixels"]
    assert (tmp_path / f"{image_name}_opened.png").is_file()
    assert (tmp_path / f"{image_name}_final.csv").is_file()
    assert (tmp_path / f"{image_name}_labels.png").is_file()
