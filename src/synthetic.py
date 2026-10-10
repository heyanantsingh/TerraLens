"""Synthetic aerial-scene generator with exact ground truth.

Real labelled before/after satellite pairs are hard to obtain offline, so the
demo and the benchmark use procedurally generated scenes. Every scene carries a
pixel-exact land-cover label map, which lets the pipeline be *measured* instead
of only eyeballed. Real imagery can be supplied through main.py instead.
"""
from dataclasses import dataclass, field
import numpy as np
import cv2

SOIL, VEG, WATER, BUILT = 0, 1, 2, 3
CLASS_NAMES = ["Bare soil", "Vegetation", "Water", "Built-up"]
N_CLASSES = 4

_BASE_RGB = np.array([[172, 140, 98], [58, 128, 64], [38, 82, 128], [0, 0, 0]], np.float32)
_LF_AMP = np.array([10, 22, 8, 0], np.float32)

FIELDS = [(20, 20, 150, 120), (200, 30, 130, 90), (30, 200, 120, 130),
          (360, 40, 120, 110), (220, 380, 150, 110)]


@dataclass
class Scene:
    lab: np.ndarray            # uint8 class map
    tone: np.ndarray           # float32 grey level used for built-up pixels
    lf: np.ndarray             # low-frequency texture field (shared between dates)
    buildings: list = field(default_factory=list)  # (x, y, w, h) rectangles

    def copy(self):
        return Scene(self.lab.copy(), self.tone.copy(), self.lf, list(self.buildings))


def _add_buildings(sc, rng, x0, x1, y0, y1, pitch=28, p=0.85):
    h, w = sc.lab.shape
    for gx in range(x0, x1 - 18, pitch):
        for gy in range(y0, y1 - 18, pitch):
            if rng.random() > p:
                continue
            bw, bh = rng.integers(14, 19, 2)
            bx, by = gx + rng.integers(0, 6), gy + rng.integers(0, 6)
            if bx <= 318 and bx + bw >= 312:      # keep clear of the vertical road
                continue
            if by <= 348 and by + bh >= 342:      # ...and the horizontal road
                continue
            if bx + bw >= w or by + bh >= h:
                continue
            t = float(rng.integers(150, 201))
            sc.lab[by:by + bh, bx:bx + bw] = BUILT
            sc.tone[by:by + bh, bx:bx + bw] = t
            sc.buildings.append((int(bx), int(by), int(bw), int(bh)))


def make_before(size=512, seed=7):
    rng = np.random.default_rng(seed)
    lab = np.zeros((size, size), np.uint8)
    tone = np.zeros((size, size), np.float32)
    coarse = rng.normal(size=(size // 32 + 2, size // 32 + 2)).astype(np.float32)
    lf = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    lf /= lf.std()
    sc = Scene(lab, tone, lf)
    for x, y, w, h in FIELDS:
        cv2.rectangle(lab, (x, y), (x + w - 1, y + h - 1), VEG, -1)
    cv2.ellipse(lab, (440, 440), (40, 30), 0, 0, 360, VEG, -1)           # forest patch
    river = np.array([[0, 170], [200, 190], [330, 250], [size - 1, 230]], np.int32)
    cv2.polylines(lab, [river], False, WATER, 16)
    cv2.ellipse(lab, (110, 430), (60, 35), 0, 0, 360, WATER, -1)          # lake
    _add_buildings(sc, rng, 226, 406, 276, 400)                           # town
    m = np.zeros((size, size), np.uint8)
    cv2.line(m, (0, 345), (size - 1, 345), 255, 6)
    cv2.line(m, (315, 0), (315, size - 1), 255, 6)
    lab[m > 0] = BUILT
    tone[m > 0] = 90.0
    return sc


def make_after(before, seed=11):
    """Apply four land-cover change events to a copy of `before`."""
    rng = np.random.default_rng(seed)
    sc = before.copy()
    lab = sc.lab
    # 1) construction: vegetated field cleared and partly built over
    reg = np.zeros_like(lab, bool); reg[30:120, 200:330] = True
    lab[reg & (lab == VEG)] = SOIL
    _add_buildings(sc, rng, 205, 310, 40, 120, pitch=28, p=0.9)
    # 2) lake shrinks (drying)
    cv2.ellipse(lab, (110, 430), (60, 35), 0, 0, 360, SOIL, -1)
    cv2.ellipse(lab, (110, 430), (36, 20), 0, 0, 360, WATER, -1)
    # 3) deforestation
    cv2.ellipse(lab, (440, 440), (40, 30), 0, 0, 360, SOIL, -1)
    # 4) agricultural expansion
    sub = lab[158:214, 380:480]
    sub[sub == SOIL] = VEG
    return sc


def render(sc, noise_seed, gain=1.0, bias=0.0, sigma=15.0):
    """Render a Scene to a BGR uint8 image with sensor noise."""
    rng = np.random.default_rng(noise_seed)
    img = _BASE_RGB[sc.lab] + (sc.lf * _LF_AMP[sc.lab])[..., None]
    built = sc.lab == BUILT
    t = sc.tone[built]
    img[built] = np.stack([t, t, t + 4], 1)
    img += rng.normal(0, sigma, img.shape).astype(np.float32)
    img = np.clip(img * gain + bias, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def translate(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def make_pair(seed=7, size=512, shift=(4, -3), gain=1.08, bias=6.0):
    """Return dict with before/after images (after is mis-registered and
    re-illuminated), exact label maps and building counts."""
    b = make_before(size, seed)
    a = make_after(b, seed + 4)
    before = render(b, seed * 100 + 1)
    after = translate(render(a, seed * 100 + 2, gain, bias), *shift)
    return dict(before=before, after=after, labels_before=b.lab, labels_after=a.lab,
                n_buildings_before=len(b.buildings), n_buildings_after=len(a.buildings),
                true_shift=shift, change_gt=(b.lab != a.lab))
