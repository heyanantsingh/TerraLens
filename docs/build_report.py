"""Builds TerraLens_Project_Report.pdf from *measured* results
(outputs/benchmark_results.json, outputs/analysis_report.json). Run from repo root."""
import json, time, os, sys
sys.path.insert(0, os.getcwd())
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether)
from src.synthetic import make_pair, CLASS_NAMES
from src.change import detect_change

NAME, REG = "Anant Singh", "24BAI10345"            
REPO = "https://github.com/heyanantsingh/TerraLens"  

B = json.load(open("outputs/benchmark_results.json"))
D = json.load(open("outputs/analysis_report.json"))
d = make_pair(7); t0 = time.time(); detect_change(d["before"], d["after"]); RUNTIME = time.time() - t0
f = lambda x, n=3: f"{x:.{n}f}"
ms = lambda p: f"{p[0]:.3f} +/- {p[1]:.3f}"

# ---- architecture figure
def arch(path):
    steps = ["Date 1 +\nDate 2 RGB", "ECC\nregistration", "Radiometric\nnormalisation", "Bilateral\ndenoise",
             "K-Means (Lab)\n+ colour rules", "Mode filter\n(land-cover map)", "Buildings, roads,\nchange mask", "Figures +\nJSON report"]
    fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off"); ax.set_xlim(0, 8.8); ax.set_ylim(0, 1)
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i * 1.1 + 0.03, 0.2), 0.98, 0.6, boxstyle="round,pad=0.02", fc="#eaf1f8", ec="#245a8d"))
        ax.text(i * 1.1 + 0.52, 0.5, s, ha="center", va="center", fontsize=7.2, weight="bold")
        if i < len(steps) - 1: ax.annotate("", xy=(i * 1.1 + 1.1, 0.5), xytext=(i * 1.1 + 1.02, 0.5), arrowprops=dict(arrowstyle="->", color="#333"))
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
arch("outputs/architecture.png")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, textColor=colors.HexColor("#245a8d"), spaceBefore=10, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, textColor=colors.HexColor("#245a8d"), spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9.5, leading=13)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8.5, textColor=colors.HexColor("#245a8d"), fontName="Helvetica-Bold")
CELL = ParagraphStyle("CELL", parent=P, fontSize=8.5, leading=11)
BUL = ParagraphStyle("BUL", parent=P, leftIndent=12, bulletIndent=2)

def para(t): return Paragraph(t, P)
def bullets(items): return [Paragraph(i, BUL, bulletText="\u2022") for i in items]
def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0d8")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")])]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#245a8d"))]
    t.setStyle(TableStyle(st))
    if header:
        for j in range(len(rows[0])): data[0][j] = Paragraph(f'<font color="white"><b>{rows[0][j]}</b></font>', CELL)
        t = Table(data, colWidths=widths, repeatRows=1); t.setStyle(TableStyle(st))
    return t
def img(path, w):
    from PIL import Image as PI
    iw, ih = PI.open(path).size; return Image(path, width=w, height=w * ih / iw)

W = 17 * cm
S = []
S += [Paragraph("TerraLens", ParagraphStyle("T", parent=ss["Title"], fontSize=26, textColor=colors.HexColor("#245a8d"))),
      Paragraph("Classical-CV Land-Cover Classification and Change Analyzer for Multi-Date Satellite Imagery", ParagraphStyle("S", parent=P, fontSize=12, alignment=1)),
      Spacer(1, 8), Paragraph("Build Your Own Project (BYOP) - Project Report", ParagraphStyle("S2", parent=P, fontSize=11, alignment=1)), Spacer(1, 10),
      table([["Course", "Computer Vision"], ["Student", NAME], ["Program", "B.Tech Computer Science Engineering, VIT Bhopal University"],
             ["Registration No.", REG], ["Repository", REPO]], [4 * cm, 13 * cm], header=False), Spacer(1, 6)]

S += [Paragraph("1. Introduction", H1),
 para("TerraLens compares two RGB images of the same area taken on different dates and reports what changed into what. "
      "It uses only classical computer-vision techniques - registration, denoising, clustering, morphology, the Hough transform and "
      "colour-difference thresholding - so it needs no training labels and no GPU. The output is a land-cover map for each date "
      "(bare soil, vegetation, water, built-up), a building count, road lines, a change mask, and a from-to transition table in hectares."),
 para("The project is a demonstration and learning implementation. Its quantitative results are measured on procedurally generated scenes "
      "with exact ground truth (Section 16); it has <b>not</b> been validated on real satellite imagery."),
 Paragraph("2. Problem Statement", H1),
 para("Change monitoring from satellite images is usually done with labelled datasets and deep networks, or by eye. Two practical problems "
      "recur. First, two acquisitions are never perfectly registered or equally illuminated, so plain image differencing mostly flags "
      "registration error and brightness drift. Second, labelled data and GPUs are often unavailable. TerraLens addresses both with a "
      "label-free pipeline that first aligns and radiometrically normalises the pair and only then compares land cover."),
 Paragraph("3. Objectives", H1)]
S += bullets(["Register two dates and correct brightness/contrast differences.",
              "Classify each date into four land-cover classes without labels.",
              "Count buildings and detect road lines from the built-up class.",
              "Detect change robustly and report transitions in hectares.",
              "Measure every claim against ground truth and against a naive baseline.",
              "Provide tests, documentation and a reproducible CLI."])

S += [Paragraph("4. Functional Requirements", H1),
 table([["Module", "Requirement"],
        ["Input", "Read two images (or generate the synthetic pair); exit with a clear message if an image cannot be read; resize date 2 if shapes differ."],
        ["Registration", "Estimate translation with ECC and warp date 2 onto date 1."],
        ["Normalisation", "Match per-channel mean/std of date 2 to date 1 using only pixels that already look similar."],
        ["Classification", "Bilateral denoise, K-Means (k=8) in Lab space, name clusters with colour rules, majority-filter the map."],
        ["Object analysis", "Buildings: morphological opening + connected components + rectangularity filter. Roads: thin remainder + probabilistic Hough."],
        ["Change", "Class-transition mask gated by Lab Delta-E, cleaned with opening and a minimum-area filter."],
        ["Output", "Five figures, analysis_report.json, transition table in hectares (user-supplied ground sample distance)."]], [3.2 * cm, 13.8 * cm]),
 Paragraph("5. Non-Functional Requirements", H1),
 table([["Requirement", "How it is addressed"],
        ["CPU only", f"No deep learning; one 512x512 pair processes in about {RUNTIME:.1f} s on the build machine (CPU, single run)."],
        ["Determinism", "All random seeds fixed; scene generation, K-Means and benchmark are repeatable."],
        ["Maintainability", "One concern per module in src/; 17 automated tests."],
        ["Graceful failure", "Unreadable input exits with an error message instead of a traceback."]], [3.5 * cm, 13.5 * cm]),
 Paragraph("6. Technology Stack", H1),
 table([["Technology", "Role"], ["Python 3.12", "Implementation language"], ["OpenCV", "Bilateral filter, ECC, morphology, connected components, Hough, Otsu, colour conversion"],
        ["NumPy", "Array maths, transition matrices"], ["scikit-learn", "K-Means clustering"], ["Matplotlib", "Figures"],
        ["Pytest", "Automated tests"], ["ReportLab", "This PDF (built by docs/build_report.py from measured JSON)"]], [3.5 * cm, 13.5 * cm])]

S += [Paragraph("7. System Architecture", H1), para("Figure 1 shows the pipeline. Each box is one function or module in src/."),
      img("outputs/architecture.png", W), Paragraph("Figure 1. TerraLens pipeline", CAP),
 Paragraph("8. Design Decisions and Rationale", H1),
 para("<b>Register, then normalise, then classify.</b> ECC maximises an enhanced correlation coefficient that is invariant to gain and bias, so it can "
      "register images whose brightness differs. Normalisation follows, using only below-median-difference pixels so that real change does not bias the correction."),
 para("<b>K-Means in Lab with rule-based naming.</b> Clustering needs no labels; naming clusters with three visible-band cues (greenness, blue dominance, low chroma) turns "
      "anonymous clusters into classes. Lab features are weighted (L x 0.6, a and b x 2.0) so hue matters more than brightness. Without this, "
      "a light-grey building cluster was merged with soil on one development seed (Section 18)."),
 para("<b>Two-condition change gate.</b> A pixel counts as changed only if its class changed <i>and</i> its Lab colour difference exceeds half the Otsu threshold. "
      "This suppresses boundary flicker where the classifier disagrees between dates but the colour did not really change."),
 para("<b>Rectangularity filter for buildings.</b> A road junction survives morphological opening as a plus-shaped blob and was counted as a building. "
      "Requiring extent (area / bounding-box area) of at least 0.8 removes it, because rectangles score about 1.0 and the junction about 0.65."),
 para("<b>Synthetic benchmark.</b> Real labelled before/after pairs were not available offline. Generated scenes give exact labels so that accuracy can be measured rather than asserted.")]

S += [Paragraph("9. Design Diagrams", H1),
      para("Workflow, use-case, sequence and class/component diagrams are in the docs/ folder as Mermaid files that GitHub renders. Summary of the use cases:"),
      table([["Actor", "Use case", "System response"],
             ["Analyst", "Run demo (--demo)", "Generate synthetic pair, run pipeline, write outputs"],
             ["Analyst", "Analyse own pair (--before --after --gsd)", "Align, classify, report change in hectares"],
             ["Analyst", "Open analysis_report.json", "Areas, transitions, building counts"],
             ["Developer", "Run pytest", "17 automated tests"]], [3 * cm, 6.5 * cm, 7.5 * cm])]

S += [Paragraph("10. Implementation Details", H1),
 table([["Path", "Purpose"], ["main.py", "CLI entry point"], ["src/synthetic.py", "Scene generator with exact ground truth"],
        ["src/preprocess.py", "Denoise, ECC alignment, radiometric normalisation"], ["src/classify.py", "Lab K-Means, rule naming, mode filter, baseline"],
        ["src/features.py", "Building blobs, road lines"], ["src/change.py", "Change detection, transitions, naive baseline"],
        ["src/evaluate.py", "Accuracy, IoU, precision, recall, F1"], ["src/figures.py, src/report.py", "Figures and JSON"],
        ["tests/test_project.py", "17 tests"], ["docs/", "Design documents and report builder"]], [5 * cm, 12 * cm]),
 Paragraph("11. Input, Processing and Output", H1),
 table([["Stage", "Input", "Processing", "Output"],
        ["Register", "Date 1, Date 2", "ECC translation, warp", "Aligned date 2, shift (dx, dy)"],
        ["Normalise", "Date 1, aligned date 2", "Robust mean/std matching", "Normalised date 2"],
        ["Classify", "Each date", "Denoise, Lab K-Means, rules, mode filter", "Label map per date"],
        ["Objects", "Label map", "Opening, components, extent filter, Hough", "Buildings, road lines"],
        ["Change", "Both label maps + Delta-E", "Transition gate, opening, min-area", "Change mask, transition table"]], [2.2 * cm, 4 * cm, 6 * cm, 4.8 * cm]),
 Paragraph("12. Output Behaviour", H1),
 para("A run writes classification.png, features_date1.png, features_date2.png, change.png, area.png and analysis_report.json to the output folder. "
      "Areas are computed as pixel count x GSD squared; the GSD is a command-line input (default 10 m/px, which is only a placeholder for the synthetic demo)."),
 Paragraph("13. Installation and Execution", H1)]
S += bullets(["git clone " + REPO + " &amp;&amp; cd TerraLens", "python -m venv venv &amp;&amp; venv\\Scripts\\activate   (Windows)", "pip install -r requirements.txt",
              "python main.py --demo", "python main.py --before a.png --after b.png --gsd 10", "pytest"])

S += [Paragraph("14. User Interaction", H1), para("Command-line only: --demo, --before, --after, --gsd, --out, --seed. No GUI."),
 Paragraph("15. Testing Approach", H1),
 para("The suite has 17 automated tests; all 17 passed on the final code. They cover: generator shape and determinism, colour rules, the mode filter, "
      "denoising, classification accuracy, superiority over the pixelwise baseline, registration accuracy, radiometric normalisation, building count, "
      "junction rejection, change quality versus the naive baseline, transition-matrix bookkeeping, mask cleaning, report keys, the CLI end to end, and the unreadable-input exit path.")]

pb = B["bld_rows"]
S += [Paragraph("16. Results and Observations", H1),
 para("<b>Protocol.</b> Rules and parameters were developed on seeds 1-3. The numbers below are for seeds 11-15, which were not used during development. "
      "Every scene has 512x512 pixels, sensor noise (sigma 15 grey levels), a 4x-3 pixel mis-registration, and a gain/bias illumination change on date 2. "
      "<b>The scene layout and the four change events are the same for every seed</b>; seeds change texture, noise and building placement. "
      "The benchmark therefore measures robustness to noise and placement, not generalisation to arbitrary geometry."),
 Paragraph("Table 1. Land-cover classification (mean +/- std over 5 held-out seeds)", CAP),
 table([["Method", "Pixel accuracy", "Mean IoU"],
        ["TerraLens (denoise + Lab K-Means + rules + mode filter)", ms(B["seg_acc_ours"]), ms(B["seg_miou_ours"])],
        ["Baseline (same colour rules, pixelwise on raw image)", ms(B["seg_acc_base"]), ms(B["seg_miou_base"])]], [9 * cm, 4 * cm, 4 * cm]),
 Spacer(1, 4), Paragraph("Table 2. Per-class IoU (mean over seeds)", CAP),
 table([["Class"] + CLASS_NAMES, ["TerraLens"] + [f(v) for v in B["iou_per_class_ours"]], ["Baseline"] + [f(v) for v in B["iou_per_class_base"]]],
       [3.4 * cm] + [3.4 * cm] * 4),
 Spacer(1, 4), Paragraph("Table 3. Change detection vs ground-truth change mask", CAP),
 table([["Method", "Precision", "Recall", "F1", "IoU"],
        ["TerraLens (align + normalise + class gate)", ms(B["chg_p_ours"]), ms(B["chg_r_ours"]), ms(B["chg_f1_ours"]), ms(B["chg_iou_ours"])],
        ["Naive (grey difference + Otsu, no alignment)", ms(B["chg_p_base"]), ms(B["chg_r_base"]), ms(B["chg_f1_base"]), ms(B["chg_iou_base"])]],
       [6.2 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm]),
 Spacer(1, 4),
 para(f"<b>Registration.</b> Mean error of the estimated shift: {B['shift_err'][0]:.2f} px (std {B['shift_err'][1]:.2f}). "
      f"<b>Buildings.</b> Predicted counts equalled the true counts on all 10 scene dates (5 seeds x 2 dates): "
      + ", ".join(f"{r['pred_before']}/{r['true_before']} then {r['pred_after']}/{r['true_after']}" for r in pb) + " (predicted/true)."),
 para("The change F1 is essentially identical across seeds because the geometry is fixed; the near-perfect value should be read as 'the pipeline "
      "handles this synthetic setting', not as an expected real-world accuracy."),
 Paragraph("Demo run (seed 7)", H2),
 para(f"Estimated shift {D['estimated_shift_px']} px (true 4, -3). Changed area {D['changed_area_ha']} ha at an assumed 10 m/px. Buildings {D['buildings']['date1']} to {D['buildings']['date2']}. "
      "Largest transitions: " + "; ".join(f"{k} {v} ha" for k, v in list(D['class_transitions_ha'].items())[:4]) + "."),
 img("outputs/classification.png", 11.5 * cm), Paragraph("Figure 2. Inputs and land-cover maps for both dates", CAP),
 img("outputs/change.png", W), Paragraph("Figure 3. Detected change (red outlines) and Lab colour-difference map", CAP),
 img("outputs/features_date2.png", 9 * cm), Paragraph("Figure 4. Buildings (red boxes) and Hough road lines (yellow) on date 2", CAP),
 img("outputs/area.png", 10 * cm), Paragraph("Figure 5. Land-cover area per date", CAP)]

S += [Paragraph("17. Limitations", H1)]
S += bullets(["<b>No real imagery.</b> All quantitative results are on synthetic scenes; real scenes add shadows, seasonal colour change, clouds, mixed pixels and sensor differences that this benchmark does not contain.",
              "<b>Fixed scene layout.</b> Only noise, texture and building placement vary between benchmark seeds.",
              "<b>Rules were developed on the same generator family</b> as the test seeds (different seeds, same generator); colour thresholds are not guaranteed to transfer to real sensors.",
              "<b>Visible bands only.</b> No near-infrared, so no NDVI; vegetation vs soil relies on greenness, which seasonal change can confound.",
              "<b>Translation-only registration.</b> Rotation, scale or terrain parallax are not handled.",
              "<b>Road output is qualitative.</b> Hough returns overlapping detections of the same road, so the raw segment count is not a road count; road length is not estimated.",
              "<b>Area in hectares depends on the GSD supplied</b> by the user; the demo's 10 m/px is a placeholder.",
              "Closely spaced buildings (gap under a few pixels) would merge into one blob and be undercounted."])
S += [Paragraph("18. Challenges Faced and Fixes", H1),
 para("<b>Easy synthetic data hid the difference between methods.</b> At sensor noise sigma 7 the pixelwise baseline already scored about 99.8% accuracy on the first scene tested, so the benchmark could not "
      "separate the methods. Noise was raised to sigma 15 (a harsher, not a tuned-to-win, setting) before measuring."),
 para("<b>Cluster merging on one seed.</b> With plain Lab features, seed 3 merged a light-grey building cluster into soil (built-up IoU 0.69, 1 building detected instead of 13). "
      "Weighting hue over brightness fixed it; all development seeds then gave built-up IoU above 0.97."),
 para("<b>Road junction counted as a building.</b> The ground-truth label map itself produced one extra blob; an extent filter removed it and a test now guards this."),
 Paragraph("19. Learnings and Key Takeaways", H1)]
S += bullets(["The naive baseline (no alignment, no normalisation, no class gating) reached change F1 of about 0.58 versus about 1.0 for the full pipeline; this comparison does not isolate which stage contributed most (no per-stage ablation was run).",
              "Unsupervised clustering needs a principled naming step; the choice of feature weights decided whether a whole class survived.",
              "A benchmark must be hard enough to separate methods; an easy one flatters everything.",
              "Report what was measured, on what data, and what was not tested."])
S += [Paragraph("20. Future Enhancements", H1)]
S += bullets(["Evaluate on a public labelled change-detection dataset (e.g. real Sentinel-2 or aerial pairs) and report the gap to the synthetic numbers.",
              "Add a near-infrared band and NDVI; add shadow and cloud masks.",
              "Affine/homography registration (ORB + RANSAC) for larger misalignment.",
              "Estimate road length by skeletonisation; vary scene geometry in the generator.",
              "Compare against a small learned model (e.g. U-Net) as an upper-bound reference."])
S += [Paragraph("21. Conclusion", H1),
 para("TerraLens shows that a carefully ordered classical pipeline - register, normalise, denoise, cluster, name, gate - can detect land-cover change without labels, and that on a "
      "harsh synthetic benchmark each stage measurably matters. Its value as a course project is the end-to-end structure, the baselines and the honest evaluation; "
      "claims about real satellite performance are explicitly left to future work."),
 Paragraph("22. References", H1)]
S += bullets(["Evangelidis, G. and Psarakis, E. 'Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization.' IEEE TPAMI, 2008.",
              "Tomasi, C. and Manduchi, R. 'Bilateral Filtering for Gray and Color Images.' ICCV, 1998.",
              "Lloyd, S. 'Least Squares Quantization in PCM.' IEEE Trans. Information Theory, 1982.",
              "Matas, J., Galambos, C. and Kittler, J. 'Robust Detection of Lines Using the Progressive Probabilistic Hough Transform.' CVIU, 2000.",
              "Otsu, N. 'A Threshold Selection Method from Gray-Level Histograms.' IEEE Trans. SMC, 1979.",
              "OpenCV documentation: https://docs.opencv.org/ ; scikit-learn documentation: https://scikit-learn.org/"])
S += [Paragraph("23. Verification Note", H1),
 para("Every number in Section 16 was produced by running the code in this repository (outputs/benchmark_results.json and outputs/analysis_report.json) and is inserted into this PDF "
      "by docs/build_report.py. The test suite result (17 passed) is from the final code. Nothing in this report was measured on real satellite data.")]

def footer(c, doc):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "TerraLens - Project Report"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}"); c.restoreState()

SimpleDocTemplate("TerraLens_Project_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                  title="TerraLens Project Report", author=NAME).build(S, onFirstPage=footer, onLaterPages=footer)
print("built TerraLens_Project_Report.pdf")
