# Use cases

| Actor | Use case | System response |
|---|---|---|
| Analyst | Run demo (`--demo`) | Generates synthetic pair, runs pipeline, writes outputs |
| Analyst | Analyse own pair (`--before --after --gsd`) | Aligns, classifies, reports change in hectares |
| Analyst | Inspect `analysis_report.json` | Areas, transitions, building counts |
| Developer | `pytest` | 17 automated tests |
| Developer | `python docs/build_report.py` | Rebuilds the PDF report from measured results |
