"""Builds report/Presentation_Phase2_DL.pdf: Phase 2 (deep learning) slides, 16:9, 8 pages.

Run report/make_dl_figures.py first, then:
    uv run --no-project --with reportlab --with pandas python report/build_presentation_phase2.py
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
DLFIG, MLFIG = ROOT / "report" / "dl_figures", ROOT / "report" / "ml_figures"
OUT = ROOT / "report" / "Presentation_Phase2_DL.pdf"

TEAM = [("Pankaj Upadhyay", "2401010317"), ("Prakhar Rawat", "2401020045"), ("Om Chimurkar", "2401010306"),
        ("Harshit Kudhial", "2401010182"), ("Vishuti Jamwal", "2401010506")]

FONT_DIR = next((ROOT / ".venv" / "lib").glob("python3*/site-packages/matplotlib/mpl-data/fonts/ttf"))
for nm, f in [("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"), ("Sans-I", "DejaVuSans-Oblique.ttf")]:
    pdfmetrics.registerFont(TTFont(nm, str(FONT_DIR / f)))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-B")

PW, PH = 960, 540
hx = colors.HexColor
NAVY, BLUE, SOFT, BG = hx("#163a6b"), hx("#2a78d6"), hx("#eaf1fb"), hx("#f3f5f9")
INK, INK2, MUTED = hx("#14161a"), hx("#525866"), hx("#8a909c")
RED, GREEN, ORANGE, VIOLET = hx("#e34948"), hx("#1baf7a"), hx("#eb6834"), hx("#6250d6")
WHITE = colors.white


def style(size, color=INK, bold=False, align=0, leading=None):
    return ParagraphStyle("s", fontName="Sans-B" if bold else "Sans", fontSize=size, textColor=color,
                          leading=leading or size * 1.34, alignment=align)


# --------------------------------------------------------------------------- results
stats = json.loads((PROC / "cleaning_stats.json").read_text())
summ = pd.read_csv(RES / "summary_macro.csv")
wil = pd.read_csv(RES / "wilcoxon_scaffold.csv")
gap = pd.read_csv(RES / "scaffold_gap_by_endpoint.csv", index_col=0).mean(axis=1)
sim = pd.read_csv(RES / "auc_by_similarity.csv").pivot(index="model", columns="bin", values="mean_roc_auc")
ensw = pd.read_csv(RES / "ensemble_weights.csv")
clean = pd.read_csv(PROC / "tox21_clean.csv")
TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
         "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
FULL_LAB = int(clean[TASKS].notna().all(axis=1).sum())


def g(split, m, col="roc_auc_mean"):
    r = summ[(summ.split == split) & (summ.model == m)]
    return None if r.empty else float(r.iloc[0][col])


def pval(a, b):
    r = wil[((wil.model_a == a) & (wil.model_b == b)) | ((wil.model_a == b) & (wil.model_b == a))].iloc[0]
    return float(r.p_value)


RF, XGB, MLP, LR = (g("scaffold", m) for m in ["RandomForest", "XGBoost", "FP-MLP", "LogReg"])
GCN, GINE, AFP = (g("scaffold", m) for m in ["GCN", "GINE-MPNN", "AttentiveFP"])
HYB, ENS = g("scaffold", "Hybrid"), g("scaffold", "Ensemble RF+GNN")
ENS_PR, FPONLY = g("scaffold", "Ensemble RF+GNN", "pr_auc_mean"), g("scaffold", "XGBoost-FPonly")
W_GRAPH = ensw[ensw.split == "scaffold"].graph_weight.mean()

# --------------------------------------------------------------------------- canvas helpers
c = canvas.Canvas(str(OUT), pagesize=(PW, PH))
c.setTitle("Molecular Toxicity Prediction: Phase 2 (Deep Learning)")
c.setAuthor("AML/DL Group Project 8")
N = [0]


def rect(x, y, w, h, fill, rad=10):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, rad, stroke=0, fill=1)


def text(t, x, y_top, w, st):
    p = Paragraph(t, st)
    _, h = p.wrap(w, 2000)
    p.drawOn(c, x, y_top - h)
    return h


def image(path, x, y, w, h):
    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    k = min(w / iw, h / ih)
    c.drawImage(ir, x + (w - iw * k) / 2, y + (h - ih * k) / 2, iw * k, ih * k, mask="auto")


def card(x, y, w, h, accent=None, fill=WHITE):
    rect(x + 1.5, y - 2, w, h, hx("#dfe3ea"), 12)
    rect(x, y, w, h, fill, 12)
    if accent:
        c.setFillColor(accent)
        c.roundRect(x, y + h - 6, w, 6, 3, stroke=0, fill=1)


def badge(x, y, label, fill, size=34, font=15):
    c.setFillColor(fill)
    c.circle(x + size / 2, y + size / 2, size / 2, stroke=0, fill=1)
    c.setFillColor(WHITE)
    c.setFont("Sans-B", font)
    c.drawCentredString(x + size / 2, y + size / 2 - font * 0.36, label)


def pill(x, y, label, fill=SOFT, color=NAVY, font=11):
    w = pdfmetrics.stringWidth(label, "Sans-B", font) + 20
    rect(x, y, w, font + 11, fill, (font + 11) / 2)
    c.setFillColor(color)
    c.setFont("Sans-B", font)
    c.drawString(x + 10, y + 6.5, label)
    return w


def bullets(x, y, w, items, size=15, color=BLUE, gapy=11):
    for t in items:
        c.setFillColor(color)
        c.circle(x + 4, y - 7, 3.2, stroke=0, fill=1)
        y -= text(t, x + 18, y, w - 18, style(size, INK2)) + gapy
    return y


def slide(title=None, num=None):
    if N[0]:
        c.showPage()
    N[0] += 1
    rect(0, 0, PW, PH, BG, 0)
    if title:
        c.setFillColor(NAVY)
        c.rect(0, PH - 64, PW, 64, stroke=0, fill=1)
        c.setFillColor(BLUE)
        c.rect(0, PH - 67, PW, 3, stroke=0, fill=1)
        if num:
            badge(30, PH - 49, num, BLUE, 34, 14)
        c.setFillColor(WHITE)
        c.setFont("Sans-B", 25)
        c.drawString(78 if num else 30, PH - 42, title)
        c.setFillColor(MUTED)
        c.setFont("Sans", 8.5)
        c.drawString(40, 14, "Molecular Toxicity Prediction · Phase 2 (Deep Learning)")
        c.drawRightString(PW - 40, 14, f"{N[0]} / 8")


# ============================================================================= 1 TITLE
slide()
rect(0, 0, PW, PH, NAVY, 0)
c.setFillColor(hx("#1f4e8c"))
c.circle(PW + 40, PH + 20, 330)
c.setFillColor(hx("#2a78d6"))
c.circle(PW + 80, PH + 60, 220)
pill(48, PH - 92, "Applied Machine Learning / Deep Learning · Group Project 8", hx("#24508f"), WHITE, 11)
text("Molecular Toxicity Prediction", 48, PH - 112, 540, style(38, WHITE, True, leading=44))
text("Phase 2: Graph Neural Networks on Tox21", 48, PH - 212, 500, style(17, hx("#c9daf3")))
y = PH - 282
c.setStrokeColor(hx("#3b5f92"))
c.line(48, y + 18, 400, y + 18)
for name, roll in TEAM:
    c.setFillColor(WHITE)
    c.setFont("Sans", 13)
    c.drawString(48, y, name)
    c.setFillColor(hx("#9fb6d9"))
    c.drawString(250, y, roll)
    y -= 23
rect(560, 150, 370, 250, WHITE, 16)
image(DLFIG / "wide_architecture.png", 572, 198, 346, 170)
c.setFillColor(INK2)
c.setFont("Sans", 10)
c.drawCentredString(745, 176, "Atoms as nodes · bonds as edges · learned features")
c.setFillColor(hx("#9fb6d9"))
c.setFont("Sans", 9)
c.drawString(48, 22, "github.com/pankaj-cod/AML_DL-Project")

# ============================================================================= 2 THE QUESTION
slide("The Question We Set Out to Answer", "1")
card(40, 268, 440, 198, BLUE)
text("Phase 1 gave us a strong baseline", 62, 448, 400, style(17, INK, True))
yy = 410
for lab, v, col in [("Random Forest", RF, GREEN), ("XGBoost", XGB, ORANGE), ("Logistic Regression", LR, BLUE)]:
    c.setFillColor(INK2)
    c.setFont("Sans", 13.5)
    c.drawString(62, yy, lab)
    rect(238, yy - 4, 160 * (v - 0.75) / 0.1, 16, col, 4)
    c.setFillColor(INK)
    c.setFont("Sans-B", 13.5)
    c.drawRightString(458, yy, f"{v:.3f}")
    yy -= 38
text("ROC-AUC on chemical families the model has <b>never seen</b>", 62, 300, 400, style(11.5, MUTED))
card(500, 268, 420, 198, VIOLET)
text("…but fingerprints are <b>frozen</b>", 522, 448, 380, style(17, INK, True))
bullets(522, 414, 380, [
    "A fixed checklist of substructures — the <b>same</b> features for all 12 endpoints.",
    "Chosen by a rule, never by the task.",
    "So: could a model <b>learn</b> better features itself?"], 13.5, VIOLET, 9)
rect(40, 74, PW - 80, 180, NAVY, 14)
pill(64, 200, "PHASE 2 QUESTION", hx("#24508f"), WHITE, 11)
text("Can a neural network that reads the molecule as a <b>graph of atoms and bonds</b> "
     "beat hand-engineered fingerprints?", 64, 182, PW - 150, style(21, WHITE, leading=29))
text("Tested where it matters: on chemistry it has never seen before.", 64, 122, PW - 150,
     style(13.5, hx("#9fb6d9")))

# ============================================================================= 3 MOLECULE AS A GRAPH
slide("How the Network Reads a Molecule", "2")
card(40, 280, PW - 80, 186)
image(DLFIG / "wide_architecture.png", 52, 290, PW - 104, 168)
steps = [("1. Build the graph", BLUE, "Every <b>atom</b> is a node with 56 features (element, charge, "
                                      "ring…). Every <b>bond</b> is an edge with 12."),
         ("2. Pass messages", GREEN, "Each atom mixes in its neighbours' information. Repeat <b>4 times</b>, "
                                     "so every atom “sees” 4 bonds around it."),
         ("3. Pool and predict", VIOLET, "Combine all atoms into one vector for the molecule, then output "
                                         "<b>12 toxicity scores</b> at once.")]
for i, (h, col, d) in enumerate(steps):
    x = 40 + i * 300
    card(x, 74, 280, 192, col)
    text(h, x + 20, 202, 240, style(16, INK, True))
    text(d, x + 20, 176, 240, style(12.5, INK2))

# ============================================================================= 4 MODELS
slide("What We Trained", "3")
models = [("GCN", BLUE, GCN, "Averages its neighbours. Simplest version — ignores bond type."),
          ("GINE-MPNN", GREEN, GINE, "Sums messages that <b>include the bond</b>. Most expressive."),
          ("AttentiveFP", VIOLET, AFP, "Uses <b>attention</b>: some neighbours count more than others.")]
for i, (name, col, score, d) in enumerate(models):
    x = 40 + i * 300
    card(x, 268, 280, 198, col)
    text(name, x + 22, 446, 170, style(18, INK, True))
    c.setFillColor(col)
    c.setFont("Sans-B", 27)
    c.drawRightString(x + 258, 424, f"{score:.3f}")
    c.setFillColor(MUTED)
    c.setFont("Sans", 9)
    c.drawRightString(x + 258, 411, "ROC-AUC")
    text(d, x + 22, 400, 236, style(13, INK2))
    rect(x + 22, 288, 236, 26, BG, 7)
    c.setFillColor(INK2)
    c.setFont("Sans", 10.5)
    c.drawString(x + 32, 296.5, "multitask: 1 network → 12 endpoints")
card(40, 74, PW - 80, 180, ORANGE)
text("Plus three controls, so the comparison is honest", 64, 212, 500, style(16, INK, True))
for i, (nm, v, d) in enumerate([
        ("FP-MLP", MLP, "deep net on the <b>same fingerprints</b> — separates “deep” from “graph”"),
        ("Hybrid", HYB, "graph + fingerprint inside one network, trained together"),
        ("Ensemble", ENS, "average the <b>rankings</b> of Random Forest and the best GNN")]):
    x = 64 + i * 290
    c.setFillColor(INK)
    c.setFont("Sans-B", 13.5)
    c.drawString(x, 166, nm)
    c.setFillColor(ORANGE)
    c.setFont("Sans-B", 13.5)
    c.drawString(x + pdfmetrics.stringWidth(nm, "Sans-B", 13.5) + 10, 166, f"{v:.3f}")
    text(d, x, 154, 262, style(11.5, INK2))

# ============================================================================= 5 THE TWO DL PROBLEMS
slide("Two Problems We Had to Solve", "4")
card(40, 268, 440, 198, RED)
text("1. 17% of labels are missing", 62, 448, 400, style(17, INK, True))
text(f"Only <b>{FULL_LAB:,} of {stats['final_molecules']:,}</b> molecules have all 12 labels.",
     62, 424, 400, style(13, INK2))
yy = 398
for lab, d, ok in [("Drop them", "loses 61% of the data", False),
                   ("Fill with 0", "invents false “non-toxic” labels", False),
                   ("Mask them out", "use every real label, invent nothing", True)]:
    col = GREEN if ok else RED
    rect(62, yy - 26, 396, 30, hx("#e6f7f0") if ok else hx("#fdecec"), 8)
    c.setFillColor(col)
    c.setFont("Sans-B", 13)
    c.drawString(76, yy - 17, "✓" if ok else "✗")
    c.setFillColor(INK)
    c.setFont("Sans-B", 12.5)
    c.drawString(96, yy - 17, lab)
    c.setFillColor(INK2)
    c.setFont("Sans", 11.5)
    c.drawString(96 + pdfmetrics.stringWidth(lab, "Sans-B", 12.5) + 10, yy - 17, f"— {d}")
    yy -= 36
rect(62, 270, 396, 26, BG, 7)
c.setFillColor(NAVY)
c.setFont("Sans-B", 12.5)
c.drawCentredString(260, 278, "loss = Σ mask · BCE  /  Σ mask")
card(500, 268, 420, 198, ORANGE)
text("2. Very few molecules are toxic", 522, 448, 380, style(17, INK, True))
text("Just <b>2.8%</b> on NR-PPAR-gamma. Always answering “non-toxic” is <b>97% accurate</b> "
     "and completely useless.", 522, 422, 380, style(13, INK2))
if (MLFIG / "waffle_ppar.png").exists():
    image(MLFIG / "waffle_ppar.png", 524, 288, 104, 104)
bullets(644, 382, 266, [
    "Weight each toxic molecule <b>≈34×</b> more in the loss.",
    "Report <b>ROC-AUC</b> and <b>PR-AUC</b> — never accuracy."], 12.5, ORANGE, 8)
rect(40, 74, PW - 80, 180, NAVY, 14)
pill(64, 196, "WHY THIS IS THE DEEP-LEARNING PART", hx("#24508f"), WHITE, 11)
text("One network predicts all 12 endpoints together. The <b>mask</b> lets it learn from every molecule "
     "even when most of its labels are missing — something 12 separate classical models cannot do.",
     64, 178, PW - 150, style(17, WHITE, leading=24))

# ============================================================================= 6 RESULTS
slide("Results on Unseen Chemistry", "5")
card(40, 100, 430, 366)
text("Mean over 12 endpoints, 3 scaffold splits", 62, 456, 390, style(11.5, MUTED))
c.setFillColor(INK)
c.setFont("Sans-B", 11)
c.drawString(62, 424, "Model")
c.drawRightString(370, 424, "ROC-AUC")
c.drawRightString(448, 424, "PR-AUC")
c.setStrokeColor(hx("#cfcdc8"))
c.setLineWidth(0.8)
c.line(62, 417, 448, 417)
yy = 396
for lab, key in [("Random Forest", "RandomForest"), ("XGBoost", "XGBoost"), ("FP-MLP", "FP-MLP"),
                 ("GCN", "GCN"), ("GINE-MPNN", "GINE-MPNN"), ("AttentiveFP", "AttentiveFP"),
                 ("Hybrid", "Hybrid"), ("Ensemble RF + GNN", "Ensemble RF+GNN")]:
    best = key in ("RandomForest", "Ensemble RF+GNN")
    if best:
        rect(58, yy - 7, 394, 28, SOFT, 6)
    c.setFillColor(INK)
    c.setFont("Sans-B" if best else "Sans", 12.5)
    c.drawString(66, yy, lab)
    c.drawRightString(370, yy, f"{g('scaffold', key):.3f}")
    c.drawRightString(448, yy, f"{g('scaffold', key, 'pr_auc_mean'):.3f}")
    yy -= 38
card(490, 300, 430, 166, GREEN)
text("The graph networks tie — they don't win", 512, 448, 390, style(16, INK, True))
text(f"GNNs reach <b>{GINE:.3f}–{AFP:.3f}</b>, level with XGBoost ({XGB:.3f}) and the FP-MLP ({MLP:.3f}). "
     "Those differences are <b>not significant</b> (p &gt; 0.3).", 512, 422, 390, style(13, INK2))
text(f"But Random Forest ({RF:.3f}) beats every GNN — and that difference <b>is</b> significant "
     f"(p = {pval('RandomForest', 'GCN'):.3f}).", 512, 372, 390, style(13, INK2))
card(490, 100, 430, 184, VIOLET)
text("Combining the two wins", 512, 266, 390, style(16, INK, True))
c.setFillColor(VIOLET)
c.setFont("Sans-B", 34)
c.drawString(512, 214, f"{ENS:.3f}")
c.drawString(664, 214, f"{ENS_PR:.3f}")
c.setFillColor(INK2)
c.setFont("Sans", 10.5)
c.drawString(512, 200, "Ensemble ROC-AUC")
c.drawString(664, 200, "Ensemble PR-AUC")
text(f"Validation gives the GNN <b>{W_GRAPH:.2f}</b> of the weight — proof it sees something the "
     f"fingerprint misses. Beats RF on 26 of 36 pairs (p = {pval('Ensemble RF+GNN', 'RandomForest'):.3f}).",
     512, 182, 390, style(12.5, INK2))

# ============================================================================= 7 GENERALISATION + WHY
slide("New Chemistry — and Why GNNs Didn't Win", "6")
card(40, 268, 420, 198)
text("Accuracy lost on unseen scaffolds", 62, 448, 380, style(15, INK, True))
image(DLFIG / "col_gap.png", 50, 276, 400, 146)
card(480, 268, 440, 198, RED)
text("The graph models lose about twice as much", 502, 448, 400, style(15, INK, True))
bullets(502, 416, 400, [
    f"Random Forest drops <b>{gap['RandomForest']:.3f}</b>; GCN and AttentiveFP drop <b>{gap['GCN']:.3f}</b>.",
    f"On <i>random</i> splits the GNNs tie Random Forest ({g('random', 'GCN'):.3f} vs "
    f"{g('random', 'RandomForest'):.3f}) — the gap appears only on new chemistry.",
    f"For truly novel molecules (similarity &lt; 0.3) everything falls to about "
    f"{sim.loc['XGBoost', '<0.3']:.2f}–{sim.loc['Ensemble RF+GNN', '<0.3']:.2f}."], 12.5, RED, 7)
for i, (h, col, d) in enumerate([
        ("Too little data", BLUE, "~6,100 training molecules. Learned features need more than that."),
        ("A very strong baseline", GREEN, f"The 217 descriptors matter: without them XGBoost falls to "
                                          f"{FPONLY:.3f}."),
        ("The control proves it", ORANGE, f"FP-MLP — deep learning on the old features — also ties, at "
                                          f"{MLP:.3f}.")]):
    x = 40 + i * 300
    card(x, 74, 280, 180, col)
    text(h, x + 20, 196, 240, style(15, INK, True))
    text(d, x + 20, 172, 240, style(12.5, INK2))

# ============================================================================= 8 CONCLUSION
slide("Conclusion", "7")
for i, (v, h, d, col) in enumerate([
        (f"{RF:.3f}", "Fingerprints still win", f"Random Forest beats every graph network on unseen "
                                                f"scaffolds (p = {pval('RandomForest', 'GCN'):.3f}).", GREEN),
        (f"{GCN:.3f}", "GNNs match XGBoost", f"GCN {GCN:.3f} vs XGBoost {XGB:.3f} — no significant "
                                             f"difference.", BLUE),
        (f"{ENS:.3f}", "Together is best", "RF + GNN beats either alone: they make different mistakes.",
         VIOLET)]):
    x = 40 + i * 300
    card(x, 268, 280, 198, col)
    c.setFillColor(col)
    c.setFont("Sans-B", 38)
    c.drawString(x + 22, 392, v)
    text(h, x + 22, 376, 240, style(16, INK, True))
    text(d, x + 22, 350, 240, style(12.5, INK2))
rect(40, 74, PW - 80, 180, NAVY, 14)
pill(64, 200, "TAKEAWAY", hx("#24508f"), WHITE, 11)
text("On a dataset this size, <b>combine</b> engineered and learned features — don't replace one with the "
     "other. And always test on scaffold splits, or the numbers flatter you.", 64, 182, PW - 150,
     style(18, WHITE, leading=25))
text("Next: pre-trained graph networks · a GNN hyper-parameter search · an applicability-domain warning.",
     64, 122, PW - 150, style(12.5, hx("#9fb6d9")))

c.save()
print("wrote", OUT, "slides:", N[0])
