# Architecture

```mermaid
flowchart LR
  A[Date 1 + Date 2 RGB] --> B[ECC registration]
  B --> C[Radiometric normalisation]
  C --> D[Bilateral denoise]
  D --> E[K-Means in Lab]
  E --> F[Colour-rule cluster naming]
  F --> G[Mode filter -> land-cover maps]
  G --> H[Buildings + roads]
  G --> I[Transitions + Delta-E gate -> change mask]
  H --> J[JSON + figures]
  I --> J
```

| Module | Responsibility |
|---|---|
| `src/synthetic.py` | Scene generator with exact ground truth (benchmark + demo) |
| `src/preprocess.py` | Denoise, ECC alignment, radiometric normalisation |
| `src/classify.py` | Lab K-Means, rule naming, mode filter, pixelwise baseline |
| `src/features.py` | Building blobs (opening + extent filter), road lines (Hough) |
| `src/change.py` | Multi-date change detection, transition matrix, naive baseline |
| `src/evaluate.py` | Accuracy / IoU / precision / recall / F1 |
| `src/figures.py`, `src/report.py` | Matplotlib figures, JSON report |
