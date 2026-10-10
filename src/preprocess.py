"""Pre-processing: denoising, registration and radiometric normalisation."""
import numpy as np
import cv2


def denoise(img_bgr):
    """Edge-preserving smoothing (two bilateral passes) to suppress sensor noise."""
    out = cv2.bilateralFilter(img_bgr, 7, 40, 5)
    return cv2.bilateralFilter(out, 7, 40, 5)


def estimate_shift(ref_bgr, tgt_bgr, iters=200):
    """Estimate a translation between two dates with ECC maximisation.
    ECC is invariant to gain/bias, so it tolerates illumination differences."""
    g1 = cv2.GaussianBlur(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    g2 = cv2.GaussianBlur(cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    cc, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
    return warp, float(cc)


def align(ref_bgr, tgt_bgr):
    """Warp `tgt` onto `ref`. Returns (aligned, (dx, dy), ecc_score)."""
    warp, cc = estimate_shift(ref_bgr, tgt_bgr)
    h, w = ref_bgr.shape[:2]
    aligned = cv2.warpAffine(tgt_bgr, warp, (w, h),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
    return aligned, (float(warp[0, 2]), float(warp[1, 2])), cc


def normalize_radiometry(ref_bgr, tgt_bgr):
    """Match per-channel mean/std of `tgt` to `ref` using only the pixels that
    already look similar (below-median difference), so real change does not
    bias the correction."""
    r = ref_bgr.astype(np.float32)
    t = tgt_bgr.astype(np.float32)
    d = np.abs(r.mean(2) - t.mean(2))
    sel = d < np.median(d)
    out = np.empty_like(t)
    for c in range(3):
        a = r[..., c][sel].std() / max(t[..., c][sel].std(), 1e-6)
        b = r[..., c][sel].mean() - a * t[..., c][sel].mean()
        out[..., c] = a * t[..., c] + b
    return np.clip(out, 0, 255).astype(np.uint8)
