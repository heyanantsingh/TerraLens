"""Multi-date change detection: align -> radiometric normalise -> classify both
dates -> keep pixels whose class changed AND whose colour changed materially."""
import numpy as np
import cv2
from .preprocess import align, normalize_radiometry, denoise
from .classify import classify
from .synthetic import N_CLASSES


def delta_e(a_bgr, b_bgr):
    la = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.sqrt(((la - lb) ** 2).sum(-1))


def clean_mask(mask, min_area=150):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[cc]


def detect_change(before, after_raw, margin=12, seed=0):
    aligned, shift, ecc = align(before, after_raw)
    after = normalize_radiometry(before, aligned)
    lb = classify(before, seed=seed)
    la = classify(after, seed=seed)
    de = delta_e(denoise(before), denoise(after))
    thr, _ = cv2.threshold(np.clip(de, 0, 255).astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = clean_mask((lb != la) & (de > 0.5 * thr))
    valid = np.zeros_like(mask)
    valid[margin:-margin, margin:-margin] = True
    mask &= valid
    trans = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    np.add.at(trans, (lb[valid], la[valid]), 1)
    return dict(after_aligned=after, labels_before=lb, labels_after=la, delta_e=de,
                change_mask=mask, valid=valid, shift=shift, ecc=ecc, transitions=trans)


def naive_change(before, after_raw, margin=12):
    """Baseline: grey-level absolute difference + Otsu, no registration/normalisation."""
    d = np.abs(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
               - cv2.cvtColor(after_raw, cv2.COLOR_BGR2GRAY).astype(np.int16)).astype(np.uint8)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = m > 0
    valid = np.zeros_like(mask)
    valid[margin:-margin, margin:-margin] = True
    return mask & valid
