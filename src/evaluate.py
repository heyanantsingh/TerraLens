"""Quantitative evaluation against synthetic ground truth."""
import numpy as np
from .synthetic import make_pair, N_CLASSES
from .classify import baseline_classify
from .features import extract_built_features
from .change import detect_change, naive_change


def seg_metrics(pred, gt, valid=None):
    if valid is None:
        valid = np.ones(gt.shape, bool)
    p, g = pred[valid].astype(int), gt[valid].astype(int)
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    np.add.at(cm, (g, p), 1)
    iou = []
    for c in range(N_CLASSES):
        tp = cm[c, c]; denom = cm[c].sum() + cm[:, c].sum() - tp
        iou.append(float(tp / denom) if denom else float("nan"))
    return dict(accuracy=float(np.trace(cm) / cm.sum()), iou=iou,
                miou=float(np.nanmean(iou)), confusion=cm)


def binary_metrics(pred, gt, valid):
    p, g = pred[valid], gt[valid]
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, iou=tp / (tp + fp + fn) if tp + fp + fn else 0.0)


def evaluate_seed(seed):
    d = make_pair(seed)
    r = detect_change(d["before"], d["after"])
    valid = r["valid"]
    out = dict(seed=seed)
    out["seg_before"] = seg_metrics(r["labels_before"], d["labels_before"], valid)
    out["seg_after"] = seg_metrics(r["labels_after"], d["labels_after"], valid)
    out["seg_baseline_before"] = seg_metrics(baseline_classify(d["before"]), d["labels_before"], valid)
    f_b = extract_built_features(r["labels_before"]); f_a = extract_built_features(r["labels_after"])
    out["buildings"] = dict(true_before=d["n_buildings_before"], pred_before=len(f_b["buildings"]),
                            true_after=d["n_buildings_after"], pred_after=len(f_a["buildings"]))
    out["change"] = binary_metrics(r["change_mask"], d["change_gt"], valid)
    out["change_baseline"] = binary_metrics(naive_change(d["before"], d["after"]), d["change_gt"], valid)
    out["shift_est"] = r["shift"]; out["shift_true"] = d["true_shift"]
    return out


def benchmark(seeds=(1, 2, 3, 4, 5)):
    return [evaluate_seed(s) for s in seeds]
