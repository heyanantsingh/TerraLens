"""Matplotlib figures used by the CLI, README and report."""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from .synthetic import CLASS_NAMES

PALETTE = ["#c9a46a", "#3b9a47", "#2b6cb0", "#7a7a7a"]
CMAP = ListedColormap(PALETTE)


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _legend(fig):
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(PALETTE, CLASS_NAMES)],
               loc="lower center", ncol=4, frameon=False)


def fig_classification(before, after_aligned, lb, la, path):
    fig, ax = plt.subplots(2, 2, figsize=(9, 9.4))
    for a, im, t in zip(ax.ravel(), [_rgb(before), _rgb(after_aligned), lb, la],
                        ["Date 1 (input)", "Date 2 (aligned + normalised)",
                         "Date 1 land cover", "Date 2 land cover"]):
        a.imshow(im, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest"); a.set_title(t, fontsize=10); a.axis("off")
    _legend(fig); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def fig_features(img, feats, path, title):
    vis = _rgb(img).copy()
    for b in feats["buildings"]:
        x, y, w, h = b["bbox"]; cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 40, 40), 2)
    for x1, y1, x2, y2 in feats["road_segments"]:
        cv2.line(vis, (x1, y1), (x2, y2), (255, 230, 0), 2)
    fig, ax = plt.subplots(figsize=(6.4, 6.4)); ax.imshow(vis); ax.axis("off")
    ax.set_title(f"{title}: {len(feats['buildings'])} buildings (red), "
                 "road lines from Hough (yellow)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_change(before, after_aligned, res, path):
    ov = _rgb(after_aligned).copy()
    m = res["change_mask"].astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnt, -1, (255, 0, 60), 2)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(_rgb(before)); ax[0].set_title("Date 1")
    ax[1].imshow(ov); ax[1].set_title("Date 2 with detected change outlined")
    im = ax[2].imshow(res["delta_e"], cmap="inferno"); ax[2].set_title("Colour change map (Lab ΔE)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_area(lb, la, gsd, path):
    n = len(CLASS_NAMES)
    ha = lambda L: np.array([(L == c).sum() for c in range(n)]) * gsd * gsd / 1e4
    b, a = ha(lb), ha(la)
    x = np.arange(n); fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - 0.2, b, 0.4, label="Date 1", color="#4a5568"); ax.bar(x + 0.2, a, 0.4, label="Date 2", color="#ed8936")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES); ax.set_ylabel(f"Area (ha, GSD={gsd:g} m/px)")
    ax.legend(frameon=False); ax.set_title("Land-cover area by date")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
