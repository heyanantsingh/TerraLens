# Requirements

## Functional
- FR1 Read two images (or generate the demo pair) and validate that they can be read.
- FR2 Estimate and remove translation between dates (ECC).
- FR3 Normalise brightness/contrast of date 2 to date 1.
- FR4 Classify each date into soil / vegetation / water / built-up without labels.
- FR5 Count buildings; detect road lines.
- FR6 Produce a change mask and a from->to transition table in hectares.
- FR7 Write figures and `analysis_report.json`.

## Non-functional
- NFR1 Runs on CPU only; one 512x512 pair in a few seconds.
- NFR2 Deterministic: fixed seeds everywhere.
- NFR3 Modular: one concern per module; each unit tested.
- NFR4 Fails with a clear message on unreadable input.
