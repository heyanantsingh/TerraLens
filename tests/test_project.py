import json, os
import numpy as np
import cv2
import pytest
from src.synthetic import make_pair, SOIL, VEG, WATER, BUILT, make_before
from src.classify import rule_classify_rgb, mode_filter, classify, baseline_classify
from src.preprocess import align, normalize_radiometry, denoise
from src.features import extract_built_features
from src.change import detect_change, naive_change, clean_mask
from src.evaluate import seg_metrics, binary_metrics
from src.report import build_report
import main as cli


@pytest.fixture(scope="module")
def pair():
    return make_pair(7)


@pytest.fixture(scope="module")
def result(pair):
    return detect_change(pair["before"], pair["after"])


def test_synthetic_pair_shapes_and_classes(pair):
    assert pair["before"].shape == (512, 512, 3) and pair["before"].dtype == np.uint8
    assert set(np.unique(pair["labels_before"])) == {SOIL, VEG, WATER, BUILT}
    assert pair["change_gt"].any() and pair["n_buildings_after"] > pair["n_buildings_before"]


def test_synthetic_is_deterministic():
    assert np.array_equal(make_before(seed=3).lab, make_before(seed=3).lab)


def test_rule_classifier_known_colours():
    rgb = np.array([[172, 140, 98], [58, 128, 64], [38, 82, 128], [150, 150, 154]])
    assert rule_classify_rgb(rgb).tolist() == [SOIL, VEG, WATER, BUILT]


def test_mode_filter_removes_isolated_pixel():
    lab = np.zeros((21, 21), np.uint8); lab[10, 10] = 1
    assert mode_filter(lab, 5)[10, 10] == 0


def test_classification_accuracy(pair, result):
    m = seg_metrics(result["labels_before"], pair["labels_before"], result["valid"])
    assert m["accuracy"] > 0.99 and m["miou"] > 0.95


def test_pipeline_beats_pixelwise_baseline(pair, result):
    ours = seg_metrics(result["labels_before"], pair["labels_before"], result["valid"])["miou"]
    base = seg_metrics(baseline_classify(pair["before"]), pair["labels_before"], result["valid"])["miou"]
    assert ours > base


def test_registration_recovers_shift(pair, result):
    dx, dy = result["shift"]
    assert abs(dx - pair["true_shift"][0]) < 0.5 and abs(dy - pair["true_shift"][1]) < 0.5


def test_radiometric_normalisation_reduces_difference(pair):
    aligned, _, _ = align(pair["before"], pair["after"])
    norm = normalize_radiometry(pair["before"], aligned)
    d0 = np.abs(pair["before"].astype(float) - aligned).mean()
    d1 = np.abs(pair["before"].astype(float) - norm).mean()
    assert d1 < d0


def test_denoise_reduces_noise(pair):
    flat = pair["before"][20:60, 20:60]
    assert denoise(pair["before"])[20:60, 20:60].std() < flat.std()


def test_building_count_close_to_truth(pair, result):
    f = extract_built_features(result["labels_before"])
    assert abs(len(f["buildings"]) - pair["n_buildings_before"]) <= 1


def test_road_junction_not_counted_as_building(pair):
    f = extract_built_features(pair["labels_before"])
    assert len(f["buildings"]) == pair["n_buildings_before"]


def test_change_detection_quality_and_baseline(pair, result):
    ours = binary_metrics(result["change_mask"], pair["change_gt"], result["valid"])
    naive = binary_metrics(naive_change(pair["before"], pair["after"]), pair["change_gt"], result["valid"])
    assert ours["f1"] > 0.95 and ours["f1"] > naive["f1"]


def test_transition_matrix_counts_all_valid_pixels(result):
    assert result["transitions"].sum() == result["valid"].sum()


def test_clean_mask_drops_small_blobs():
    m = np.zeros((60, 60), bool); m[5:8, 5:8] = True; m[20:50, 20:50] = True
    out = clean_mask(m)
    assert not out[6, 6] and out[30, 30]


def test_report_keys(result):
    f = extract_built_features(result["labels_before"])
    rep = build_report(result, f, f, 10.0, ["x"])
    assert {"area_ha_date1", "changed_area_ha", "class_transitions_ha", "buildings"} <= set(rep)


def test_cli_demo_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rep = cli.main(["--demo", "--out", str(tmp_path)])
    assert (tmp_path / "analysis_report.json").exists() and (tmp_path / "change.png").exists()
    assert json.load(open(tmp_path / "analysis_report.json"))["changed_area_ha"] == rep["changed_area_ha"]


def test_cli_missing_image_exits():
    with pytest.raises(SystemExit):
        cli.main(["--before", "nope.png", "--after", "nope2.png"])
