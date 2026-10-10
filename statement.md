# Problem Statement

Land-cover monitoring (urban growth, deforestation, shrinking water bodies, farmland expansion) is normally done
with large labelled datasets and deep-learning models, or by manually comparing satellite images. Two practical
obstacles remain: (1) two acquisitions of the same area are never perfectly registered or equally illuminated, so
naive image differencing produces mostly false alarms; and (2) many students and small teams have no labelled data
and no GPU.

**TerraLens** is a classical computer-vision pipeline that takes two RGB images of the same area taken on different
dates and, without any training labels, (a) registers and radiometrically normalises them, (b) classifies each date
into bare soil / vegetation / water / built-up, (c) counts buildings and finds road lines, and (d) reports *what
changed into what* in hectares.

**Scope.** Visible-band imagery only (no NIR), single-sensor pairs, translation-level mis-registration. Quantitative
results in this repository are measured on procedurally generated scenes with exact ground truth; see README for
the limits of that claim.
