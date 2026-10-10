"""Unsupervised land-cover classification: K-Means in CIELAB space, then
clusters are named by physically-motivated colour rules (no training labels)."""
import numpy as np
import cv2
from sklearn.cluster import KMeans
from .synthetic import SOIL, VEG, WATER, BUILT, N_CLASSES
from .preprocess import denoise


def rule_classify_rgb(rgb):
    """Name colours (R,G,B, shape (...,3)) using three visible-band cues:
    greenness (veg), blue dominance (water), low chroma (built-up/roads)."""
    rgb = np.asarray(rgb, np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = rgb.max(-1) - rgb.min(-1)
    out = np.full(r.shape, SOIL, np.uint8)
    out[chroma < 35] = BUILT
    out[(g - np.maximum(r, b)) > 10] = VEG
    out[(b - np.maximum(r, g)) > 18] = WATER
    return out


def mode_filter(labels, ksize=5):
    """Majority vote in a ksize x ksize window (removes salt-and-pepper)."""
    votes = np.stack([cv2.boxFilter((labels == c).astype(np.float32), -1, (ksize, ksize))
                      for c in range(N_CLASSES)], -1)
    return votes.argmax(-1).astype(np.uint8)


def baseline_classify(img_bgr):
    """Naive baseline: the same colour rules applied pixel-by-pixel on the raw image."""
    return rule_classify_rgb(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def classify(img_bgr, k=8, seed=0, mode_ksize=5):
    """Full pipeline: denoise -> Lab K-Means -> rule naming -> majority filter."""
    sm = denoise(img_bgr)
    rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB).astype(np.float32)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    X = lab.reshape(-1, 3) * np.float32([0.6, 2.0, 2.0])   # emphasise hue over brightness
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(20000, len(X)), replace=False)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X[idx])
    assign = km.predict(X).reshape(h, w)
    cluster_class = np.zeros(k, np.uint8)
    for c in range(k):
        m = assign == c
        if m.any():
            cluster_class[c] = rule_classify_rgb(rgb[m].mean(0))
    return mode_filter(cluster_class[assign], mode_ksize)
