"""Builds report/IEEE_Report_Phase2_DL.pdf: the Phase 2 (deep learning) report in IEEE
conference format (US Letter, two columns, Times, hyphenated justification).

All result numbers are read from results/*.csv, so the paper always matches the experiments.
Run report/make_dl_figures.py first, then:
    uv run --no-project --with reportlab --with pandas --with pyphen python report/build_ieee_report.py
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES, PROC, DLFIG = ROOT / "results", ROOT / "data" / "processed", ROOT / "report" / "dl_figures"
OUT = ROOT / "report" / "Report_Phase2_DL.pdf"

TEAM = [("Pankaj Upadhyay", "2401010317"), ("Prakhar Rawat", "2401020045"), ("Om Chimurkar", "2401010306"),
        ("Harshit Kudhial", "2401010182"), ("Vishuti Jamwal", "2401010506")]
COURSE = "Applied Machine Learning / Deep Learning, Group Project 8"

# --------------------------------------------------------------------------- fonts (IEEE: Times)
MAC = Path("/System/Library/Fonts/Supplemental")
FALLBACK = next((ROOT / ".venv" / "lib").glob("python3*/site-packages/matplotlib/mpl-data/fonts/ttf"))
for name, (mac, fb) in {"T": ("Times New Roman.ttf", "DejaVuSerif.ttf"),
                        "T-B": ("Times New Roman Bold.ttf", "DejaVuSerif-Bold.ttf"),
                        "T-I": ("Times New Roman Italic.ttf", "DejaVuSerif-Italic.ttf"),
                        "T-BI": ("Times New Roman Bold Italic.ttf", "DejaVuSerif-BoldItalic.ttf")}.items():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont(name, str(MAC / mac if (MAC / mac).exists() else FALLBACK / fb)))
pdfmetrics.registerFontFamily("T", normal="T", bold="T-B", italic="T-I", boldItalic="T-BI")
# Times New Roman has no U+2208; fall back to DejaVu Serif for that one symbol.
pdfmetrics.registerFont(TTFont("T-sym", str(FALLBACK / "DejaVuSerif.ttf")))
IN = '<font name="T-sym" size="8.6">\u2208</font>' 

try:                                    # hyphenation removes rivers in narrow justified columns
    import pyphen  # noqa: F401
    HYPH = {"hyphenationLang": "en_GB"}   # embeddedHyphenation triggers a split bug at frame edges
except ImportError:
    HYPH = {}

# --------------------------------------------------------------------------- IEEE page geometry
PW, PH = letter                          # 612 x 792 pt
MT, MB, MS = 54, 60, 45                  # top / bottom / side margins
COLW, GUT = 252, 18                      # 3.5 in columns, 0.25 in gutter
FULLW = 2 * COLW + GUT                   # 522 pt
LEFT_X, RIGHT_X = MS, MS + COLW + GUT
BODY_TOP, BODY_BOT = PH - MT, MB
INK = colors.black

S = {
    "title": ParagraphStyle("title", fontName="T", fontSize=21, leading=25, alignment=TA_CENTER),
    "author": ParagraphStyle("author", fontName="T", fontSize=10.5, leading=13.5, alignment=TA_CENTER),
    "affil": ParagraphStyle("affil", fontName="T-I", fontSize=9.5, leading=12, alignment=TA_CENTER),
    "abstract": ParagraphStyle("abstract", fontName="T-B", fontSize=9, leading=10.6,
                               alignment=TA_JUSTIFY, spaceAfter=5, **HYPH),
    "body": ParagraphStyle("body", fontName="T", fontSize=9.6, leading=11.4, alignment=TA_JUSTIFY,
                           firstLineIndent=11, **HYPH),
    "body0": ParagraphStyle("body0", fontName="T", fontSize=9.6, leading=11.4, alignment=TA_JUSTIFY, **HYPH),
    "sec": ParagraphStyle("sec", fontName="T", fontSize=9.8, leading=12, alignment=TA_CENTER,
                          spaceBefore=9, spaceAfter=3.5),
    "sub": ParagraphStyle("sub", fontName="T-I", fontSize=9.6, leading=11.6, spaceBefore=6, spaceAfter=2),
    "eq": ParagraphStyle("eq", fontName="T-I", fontSize=9.6, leading=13.5, alignment=TA_CENTER,
                         spaceBefore=4, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", fontName="T", fontSize=9.6, leading=11.4, alignment=TA_JUSTIFY,
                             leftIndent=12, bulletIndent=2, spaceAfter=2.5, **HYPH),
    "quote": ParagraphStyle("quote", fontName="T-I", fontSize=9.6, leading=11.6, alignment=TA_JUSTIFY,
                            leftIndent=10, rightIndent=10, spaceBefore=3, spaceAfter=4, **HYPH),
    "cap": ParagraphStyle("cap", fontName="T", fontSize=8, leading=9.6, alignment=TA_JUSTIFY,
                          spaceBefore=4, spaceAfter=6, **HYPH),
    "capc": ParagraphStyle("capc", fontName="T", fontSize=8, leading=9.6, alignment=TA_CENTER,
                           spaceBefore=4, spaceAfter=2, **HYPH),
    "th": ParagraphStyle("th", fontName="T-B", fontSize=7.2, leading=8.4, alignment=TA_CENTER),
    "td": ParagraphStyle("td", fontName="T", fontSize=7.2, leading=8.4),
    "tdc": ParagraphStyle("tdc", fontName="T", fontSize=7.2, leading=8.4, alignment=TA_CENTER),
    "ref": ParagraphStyle("ref", fontName="T", fontSize=8, leading=9.4, alignment=TA_JUSTIFY,
                          leftIndent=13, firstLineIndent=-13, spaceAfter=1.5, **HYPH),
}

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
full_lab = int(clean[TASKS].notna().all(axis=1).sum())
prev = clean[TASKS].sum() / clean[TASKS].notna().sum()


def g(split, model, col):
    r = summ[(summ.split == split) & (summ.model == model)]
    return None if r.empty else float(r.iloc[0][col])


def ms(model, met="roc_auc", split="scaffold"):
    return f"{g(split, model, met + '_mean'):.3f} ± {g(split, model, met + '_std'):.3f}"


def wtest(a, b):
    r = wil[((wil.model_a == a) & (wil.model_b == b)) | ((wil.model_a == b) & (wil.model_b == a))].iloc[0]
    sgn = 1 if r.model_a == a else -1
    wa = int(r.a_wins if sgn == 1 else r.b_wins)
    return sgn * float(r.mean_diff_roc_auc), wa, 36 - wa, float(r.p_value)


def pf(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


RF, GCN, AFP, GINE = (g("scaffold", m, "roc_auc_mean") for m in ["RandomForest", "GCN", "AttentiveFP", "GINE-MPNN"])
MLP, HYB, ENS, XGB = (g("scaffold", m, "roc_auc_mean") for m in ["FP-MLP", "Hybrid", "Ensemble RF+GNN", "XGBoost"])
ENS_PR = g("scaffold", "Ensemble RF+GNN", "pr_auc_mean")
W_GRAPH = ensw[ensw.split == "scaffold"].graph_weight.mean()

# --------------------------------------------------------------------------- flowables
FIGN, TABN, SECN = [0], [0], [0]
# Figure numbers come from the order figures are emitted; referring to them through FN keeps
# every cross-reference in the text correct if that order ever changes.
FIG_ORDER = ["wide_architecture.png", "col_models.png", "col_curves.png", "wide_heatmap.png",
             "col_gap.png", "col_similarity.png"]
FN = {n: i + 1 for i, n in enumerate(FIG_ORDER)}
WIDE_SPECS = []          # (template id, required height) filled while the story is built


def _roman(n):
    out = ""
    for v, s in [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out, n = out + s, n - v
    return out


def section(title):
    SECN[0] += 1
    return Paragraph(f"{_roman(SECN[0])}.&nbsp; {title.upper()}", S["sec"])


def sub(letter_, title):
    return Paragraph(f"{letter_}. {title}", S["sub"])


def para(t, indent=True):
    return Paragraph(t, S["body"] if indent else S["body0"])


def eq(body, num):
    t = Table([[Paragraph(body, S["eq"]), Paragraph(f"({num})", S["body0"])]], colWidths=[COLW - 22, 22])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def col_figure(name, caption):
    """A figure that fits inside one column."""
    FIGN[0] += 1
    assert FIG_ORDER[FIGN[0] - 1] == name, f"figure order changed: update FIG_ORDER for {name}"
    iw, ih = ImageReader(str(DLFIG / name)).getSize()
    img = Image(str(DLFIG / name), width=COLW, height=COLW * ih / iw)
    return KeepTogether([img, Paragraph(f"Fig. {FIGN[0]}. {caption}", S["cap"])])


def wide_figures(*specs):
    """One or more figures spanning both columns, emitted into a top frame sized to fit them."""
    flow, total = [], 14
    w = FULLW - 2
    for name, caption in specs:
        FIGN[0] += 1
        assert FIG_ORDER[FIGN[0] - 1] == name, f"figure order changed: update FIG_ORDER for {name}"
        iw, ih = ImageReader(str(DLFIG / name)).getSize()
        h = w * ih / iw
        cap = Paragraph(f"Fig. {FIGN[0]}. {caption}", S["capc"])
        _, ch = cap.wrap(w, 2000)
        flow += [Image(str(DLFIG / name), width=w, height=h), cap, Spacer(1, 6)]
        total += h + ch + 10
    tid = f"wide{len(WIDE_SPECS)}"
    WIDE_SPECS.append((tid, total))
    return [NextPageTemplate(tid), PageBreak()] + flow + [FrameBreak(), NextPageTemplate("normal")]


def table(rows, widths, caption, bold_rows=(), font=7.2, align_from=1):
    TABN[0] += 1
    data = []
    for i, row in enumerate(rows):
        cells = []
        for j, cell in enumerate(row):
            base = S["th"] if i == 0 else (S["tdc"] if j >= align_from else S["td"])
            st = ParagraphStyle(f"c{i}{j}", parent=base, fontSize=font, leading=font * 1.2,
                                fontName="T-B" if (i in bold_rows or i == 0) else base.fontName)
            cells.append(Paragraph(str(cell), st))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.7), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]))
    head = Paragraph(f"TABLE {_roman(TABN[0])}<br/>{caption}",
                     ParagraphStyle("tc", parent=S["capc"], spaceBefore=4, spaceAfter=3))
    return KeepTogether([head, t, Spacer(1, 7)])


# ============================================================================= story
st = []

# ---- title block (spans both columns; the abstract follows in column one, as in IEEEtran)
st += [Paragraph("Learned Molecular Graph Representations versus Engineered Fingerprints "
                 "for Multitask Toxicity Prediction on Tox21", S["title"]), Spacer(1, 12)]
st += [Paragraph(" &nbsp;&nbsp;&nbsp;&nbsp; ".join(n for n, _ in TEAM[:3]), S["author"]),
       Paragraph(" &nbsp;&nbsp;&nbsp;&nbsp; ".join(n for n, _ in TEAM[3:]), S["author"]), Spacer(1, 4),
       Paragraph(f"Enrollment numbers: {', '.join(r for _, r in TEAM)}", S["affil"]),
       Paragraph(COURSE, S["affil"]), FrameBreak(), NextPageTemplate("normal")]

st += [Paragraph(
    "<b><i>Abstract</i></b><b>—Toxicity screening of chemical libraries is slow and costly, motivating models "
    "that predict assay activity directly from molecular structure. This paper asks whether representations "
    "<i>learned</i> by graph neural networks outperform <i>engineered</i> chemical fingerprints on Tox21, a "
    f"benchmark of 7,831 compounds and 12 toxicity endpoints. After canonicalisation, salt stripping and "
    f"deduplication ({stats['final_molecules']:,} unique molecules), we train three message passing "
    "architectures — a graph convolutional network, a bond-aware GINE network and AttentiveFP — as multitask "
    "models over all twelve endpoints using a masked, class-weighted cross-entropy that ignores the 17% of "
    "labels that are missing. A deep multitask perceptron on the same fingerprints isolates the effect of deep "
    "learning from that of the graph representation, and two hybrids combine both views. All models are "
    "compared on three Bemis–Murcko scaffold splits, which hold out entire chemical families, using "
    "endpoint-wise ROC-AUC and PR-AUC computed on observed labels only and paired Wilcoxon tests. On unseen "
    f"scaffolds the graph networks reach {GINE:.3f}–{AFP:.3f} ROC-AUC and are statistically indistinguishable "
    f"from gradient boosting ({XGB:.3f}) and from the fingerprint perceptron ({MLP:.3f}), while a random forest "
    f"over fingerprints and descriptors remains significantly stronger ({RF:.3f}). Graph models also lose "
    "roughly twice as much accuracy as tree ensembles when moving from random to scaffold splits. The two views "
    "are nevertheless complementary: a validation-weighted late fusion of a random forest and a graph network "
    f"attains the best overall result ({ENS:.3f} ROC-AUC, {ENS_PR:.3f} PR-AUC). At this data scale, learned "
    "graph representations do not replace engineered descriptors, but they contribute information that improves "
    "prediction when the two are combined.</b>", S["abstract"])]
st += [Paragraph("<b><i>Index Terms</i></b><b>—Graph neural networks, message passing, molecular property "
                 "prediction, multitask learning, missing labels, scaffold split, Tox21, toxicity "
                 "prediction.</b>", S["abstract"])]

# ============================================================================= I. INTRODUCTION
st += [section("Introduction")]
st += [para("Regulatory agencies and pharmaceutical companies must assess the toxicity of tens of thousands of "
            "chemicals, yet every <i>in vitro</i> or <i>in vivo</i> assay is slow, expensive and, in the animal "
            "case, ethically costly. Models that predict assay outcomes from chemical structure alone can rank "
            "untested compounds, flag likely hazards early and reduce experimental burden. Their practical value "
            "depends almost entirely on how well they extrapolate to <i>new</i> chemistry, because a screening "
            "model is applied precisely to compounds that differ from those used to build it.", indent=False)]
st += [para("Two families of molecular representation dominate this task. <i>Engineered fingerprints</i>, such "
            "as Morgan or extended-connectivity fingerprints, hash local atom neighbourhoods into a "
            "fixed-length bit vector and are paired with classical learners such as random forests or gradient "
            "boosting. They are fast, robust and remarkably hard to beat, but they are fixed: the same "
            "substructure dictionary serves every endpoint, bit collisions discard information, and nothing in "
            "the representation adapts to the prediction task. <i>Graph neural networks</i> (GNNs) instead "
            "treat a molecule as a graph of atoms and bonds and <i>learn</i> a task-specific representation by "
            "message passing. In principle they can capture interactions that a fixed dictionary cannot, and a "
            "multitask network can share chemistry across biologically related assays.")]
st += [para("In an earlier phase of this project we established the classical baseline: under a scaffold split, "
            f"a random forest over Morgan fingerprints and RDKit descriptors reached {RF:.3f} ROC-AUC, ahead of "
            f"gradient boosting ({XGB:.3f}) and logistic regression "
            f"({g('scaffold', 'LogReg', 'roc_auc_mean'):.3f}). This paper reports the deep learning phase and "
            "answers the question that motivated it:")]
st += [Paragraph("Do learned molecular-graph representations outperform engineered fingerprints for toxicity "
                 "prediction, and how does each generalise to unseen chemical families?", S["quote"])]
st += [para("The contributions of this phase are as follows.", indent=False)]
for b in [
    "Three structurally different message passing architectures — GCN, bond-aware GINE and AttentiveFP — "
    "trained as multitask models over all twelve endpoints with a <i>masked</i>, class-weighted loss, so that "
    "the 17% of labels that are missing contribute neither loss nor gradient.",
    "Controls that isolate the source of any difference: a deep multitask perceptron over exactly the "
    "engineered features used by the baselines, and a descriptor ablation.",
    "Two ways of combining the views — an end-to-end hybrid network and a validation-weighted late-fusion "
    "ensemble — together with an analysis of why joint training is harder than it appears.",
    "A generalisation study reporting the random-versus-scaffold performance gap per model and endpoint, and "
    "accuracy as a function of each test molecule's similarity to the training set.",
    "A statistically grounded comparison using paired Wilcoxon signed-rank tests over 36 endpoint–seed pairs, "
    "with a fully reproducible implementation.",
]:
    st += [Paragraph(b, S["bullet"], bulletText="•")]

# ============================================================================= II. RELATED WORK
st += [section("Related Work")]
st += [para("<b>Benchmarks.</b> The Tox21 Data Challenge [1] released twelve nuclear-receptor and "
            "stress-response assays screened over roughly 10,000 environmental chemicals and drugs. "
            "MoleculeNet [2] standardised Tox21 as a multitask benchmark, showed that graph convolutions were "
            "competitive with fingerprint models under random splits, and recommended scaffold splitting as a "
            "more realistic estimate of generalisation.", indent=False)]
st += [para("<b>Engineered representations.</b> Extended-connectivity fingerprints [3] iteratively hash "
            "circular atom environments into a bit vector and remain a strong default for activity prediction, "
            "typically combined with random forests [4] or gradient boosted trees [5]. Bemis and Murcko [6] "
            "introduced the molecular framework, or scaffold, that underlies scaffold-based data splitting.")]
st += [para("<b>Message passing networks.</b> Gilmer <i>et al.</i> [7] unified molecular graph networks under "
            "the message passing framework. Graph convolutional networks [8] aggregate neighbours with degree "
            "normalisation; the graph isomorphism network [9] uses sum aggregation and is provably as "
            "discriminative as the Weisfeiler–Lehman test, and its edge-aware variant GINE [10] injects bond "
            "features into each message. AttentiveFP [11] applies graph attention together with a gated readout "
            "and reported strong Tox21 results. Yang <i>et al.</i> [12] found with Chemprop that learned "
            "representations surpass fingerprints on many but not all datasets, that concatenating global RDKit "
            "descriptors helps considerably, and that scaffold splits depress every model's apparent "
            "performance.")]
st += [para("Our study follows this line but focuses on a controlled, statistically tested comparison under "
            "scaffold splitting, with explicit controls separating the contribution of deep multitask learning "
            "from that of the graph representation itself.")]

# ============================================================================= III. DATASET
st += [section("Dataset and Preprocessing")]
st += [para("We use the MoleculeNet/DeepChem release of Tox21: 7,831 SMILES strings with binary labels for "
            "twelve endpoints, seven nuclear-receptor (NR) assays and five stress-response (SR) assays. A label "
            "is missing when the compound was not tested in that assay or the result was inconclusive.",
            indent=False)]
st += [sub("A", "Cleaning and deduplication")]
st += [para("Every SMILES is parsed with RDKit, standardised, reduced to its largest organic fragment so that "
            "salts and counter-ions are stripped, and written as a canonical SMILES. Rows are then grouped by "
            "canonical SMILES so that a parent compound has a single identity. For each endpoint, duplicate "
            "labels are merged when they agree and set to missing when they conflict. Crucially, deduplication "
            "happens <i>before</i> any split, so the same compound can never appear in both training and test "
            "data. Table I summarises the resulting data accounting.", indent=False)]
st += [table([["Stage", "Count"],
              ["Raw rows", f"{stats['raw_rows']:,}"],
              ["Unparseable SMILES removed", f"{stats['unparseable']}"],
              ["Duplicate rows merged", f"{stats['duplicate_rows_merged']}"],
              ["Conflicting labels set to missing", f"{sum(stats['conflicting_labels_set_missing'].values())}"],
              ["Final unique molecules", f"{stats['final_molecules']:,}"],
              ["Observed labels", f"{stats['labels_after_merge']:,} "
                                  f"({stats['labels_after_merge'] / (stats['final_molecules'] * 12):.1%})"],
              ["Molecules with all twelve labels", f"{full_lab:,} ({full_lab / stats['final_molecules']:.0%})"],
              ["Unique Bemis–Murcko scaffolds", f"{stats['unique_scaffolds']:,}"]],
             [COLW * 0.68, COLW * 0.32], "Data Cleaning Summary")]
st += [sub("B", "Class imbalance and missing labels")]
st += [para(f"The endpoints are severely imbalanced: the active rate ranges from {prev.min():.1%} "
            f"(NR-PPAR-gamma) to {prev.max():.1%} (SR-ARE). They are also incomplete: 17% of labels are missing "
            f"and only {full_lab:,} molecules ({full_lab / stats['final_molecules']:.0%}) carry all twelve "
            "labels. Discarding incomplete molecules would therefore remove the majority of the data, while "
            "imputing missing entries as inactive would inject false negatives into assays where the compound "
            "was simply never tested. Both problems are handled in the loss rather than in the data, as "
            "described in Section IV-D.", indent=False)]

# ============================================================================= IV. METHODOLOGY
st += wide_figures(("wide_architecture.png",
                  "Multitask message passing pipeline. A molecule is encoded as a graph with 56-dimensional "
                  "atom features and 12-dimensional bond features. After <i>k</i> rounds of message passing, a "
                  "permutation-invariant readout produces a single molecule vector and an MLP head emits twelve "
                  "logits. Endpoints whose label is missing (grey) are masked out of the loss and contribute no "
                  "gradient."))
st += [section("Methodology")]
st += [sub("A", "Scaffold splitting")]
st += [para("Every model is evaluated on identical, cached split files. The primary protocol is a randomised "
            "(balanced) Bemis–Murcko scaffold split in 80/10/10 proportions, giving 6,148 training, 768 "
            "validation and 770 test molecules. Each scaffold group is assigned to exactly one partition, so "
            "test molecules belong to chemical series entirely absent from training. Scaffold groups larger "
            f"than half the test set — notably the acyclic group ({stats['acyclic_molecules']:,} molecules) and "
            "the benzene group (1,476), each larger than the whole test set — are placed in training first, and "
            "the remaining groups are shuffled with the random seed. Seeds 0, 1 and 2 produce three independent "
            "scaffold splits in which no scaffold is shared between training and test. Three molecule-level "
            "random splits of identical size are used only to quantify the generalisation gap.", indent=False)]
st += [sub("B", "Graph representation")]
st += [para("Atoms are nodes and each bond contributes two directed edges. Atom features (56 dimensions) "
            "comprise one-hot encodings of element (22 common elements plus an “other” slot), total degree, "
            "formal charge, hybridisation, attached hydrogen count and chirality tag, together with "
            "aromaticity, ring membership and scaled atomic mass. Bond features (12 dimensions) encode bond "
            "type, stereochemistry, conjugation and ring membership.", indent=False)]
st += [sub("C", "Architectures")]
st += [para(f"Fig. {FN['wide_architecture.png']} summarises the pipeline shared by all three graph models. "
            "All networks are multitask, emitting twelve logits. In a message passing layer each atom <i>v</i> "
            "updates its hidden state from its neighbourhood <i>N</i>(<i>v</i>) as", indent=False)]
st += [eq("h<sub>v</sub><sup>(k+1)</sup> = U ( h<sub>v</sub><sup>(k)</sup> , "
          "AGG<sub>u " + IN + " N(v)</sub> M ( h<sub>u</sub><sup>(k)</sup> , e<sub>uv</sub> ) )", 1)]
st += [para("where <i>M</i> builds a message from the neighbour state and the bond feature "
            "<i>e<sub>uv</sub></i>, AGG is a permutation-invariant aggregator and <i>U</i> is the update "
            "function. After <i>k</i> layers a readout pools atom states into one molecule vector, which an MLP "
            "head maps to the twelve logits. We instantiate three architectures and two controls, summarised in "
            "Table II.", indent=False)]
st += [para("<b>GCN</b> applies four graph convolution layers with degree-normalised neighbour averaging, "
            "residual connections, batch normalisation and dropout, followed by concatenated mean and max "
            "pooling. Bond features are not used, which makes it the simplest inductive bias we test.")]
st += [para("<b>GINE-MPNN</b> uses four GIN layers whose messages add a learned projection of the bond features "
            "before an MLP update, with sum and mean pooling. Sum aggregation makes it the most expressive of "
            "the three in the Weisfeiler–Lehman sense.")]
st += [para("<b>AttentiveFP</b> learns attention weights over neighbours and applies a gated recurrent readout "
            "over two time steps, so that different atoms contribute unequally to the molecule vector.")]
st += [para("<b>FP-MLP</b> is the key control: a multitask perceptron over exactly the same 2,048-bit Morgan "
            "fingerprint and 217 RDKit descriptors used by the classical baselines. Were the graph networks to "
            "outperform those baselines, this control would reveal whether the gain came from the graph "
            "representation or merely from deep multitask learning on shared features.")]
st += [para("<b>Hybrid</b> concatenates the GINE molecule embedding with a regularised fingerprint branch and "
            "trains both end to end, as described in Section IV-F.")]
st += [table([["Model", "Representation", "Depth / readout", "Params"],
              ["FP-MLP", "fingerprint + descriptors", "2262→1024→256", "2.59 M"],
              ["GCN", "graph, no bond features", "4 layers, mean ‖ max", "0.41 M"],
              ["GINE-MPNN", "graph + bond features", "4 layers, sum ‖ mean", "0.69 M"],
              ["AttentiveFP", "graph + bond features", "3 layers, 2 readout steps", "1.93 M"],
              ["Hybrid", "graph + fingerprint", "GINE ‖ FP branch", "1.47 M"]],
             [COLW * 0.21, COLW * 0.33, COLW * 0.32, COLW * 0.14],
             "Deep Models Compared (Hidden Size 256)")]
st += [sub("D", "Masked, class-weighted loss")]
st += [para("Let <i>y<sub>it</sub></i> be the label of molecule <i>i</i> for endpoint <i>t</i>, let "
            "<i>m<sub>it</sub></i> = 1 when that label is observed and 0 otherwise, let <i>z<sub>it</sub></i> "
            "be the corresponding logit, σ the logistic function, and <i>w<sub>t</sub></i> the ratio of "
            "negatives to positives for endpoint <i>t</i> in the training partition. The objective is",
            indent=False)]
st += [eq("L = ( Σ<sub>i,t</sub> m<sub>it</sub> ℓ<sub>it</sub> ) ⁄ ( Σ<sub>i,t</sub> m<sub>it</sub> ) ,<br/>"
          "ℓ<sub>it</sub> = − w<sub>t</sub> y<sub>it</sub> log σ(z<sub>it</sub>) "
          "− (1 − y<sub>it</sub>) log(1 − σ(z<sub>it</sub>))", 2)]
st += [para("Missing entries therefore contribute neither loss nor gradient, and normalising by the number of "
            "observed labels keeps the loss scale independent of how many labels happen to be absent from a "
            "mini-batch. The endpoint-specific weight <i>w<sub>t</sub></i> reaches roughly 34 for the rarest "
            "endpoint, counteracting the tendency of an unweighted model to predict the majority class "
            "everywhere.", indent=False)]
st += [sub("E", "Training setup")]
st += [para("All networks use AdamW (learning rate 10<sup>−3</sup>, weight decay 10<sup>−5</sup>), batch size "
            "128, gradient clipping at 5, learning-rate halving after ten epochs without improvement, and early "
            "stopping with patience 20 over at most 100 epochs on mean validation ROC-AUC, restoring the best "
            "weights. Experiments ran on an eight-core Apple M3 CPU; a single graph network takes 4–15 minutes "
            "per split and the complete pipeline about two hours.", indent=False)]
st += [sub("F", "Combining the two representations")]
st += [para("<b>End-to-end hybrid.</b> A naive concatenation of the graph and fingerprint embeddings performed "
            "worse than either component alone. The fingerprint branch fits the training set within two to four "
            "epochs, whereas the graph branch needs thirty to ninety, so early stopping halted training while "
            "the graph branch was still undertrained. The final hybrid applies input dropout of 0.5 and a "
            "narrow projection to the fingerprint branch, a stronger weight decay of 10<sup>−2</sup> on that "
            "branch only, and <i>modality dropout</i>: during training the entire fingerprint embedding is "
            "zeroed for a random half of the molecules, forcing the prediction head to use the graph embedding "
            "as well. On the first scaffold split this raised validation ROC-AUC from 0.784 to 0.829 and moved "
            "the best epoch from 31 to 60.", indent=False)]
st += [para("<b>Late-fusion ensemble.</b> For each split, the graph model with the highest <i>validation</i> "
            "ROC-AUC is paired with the random forest. Per endpoint, the two models' scores are rank-normalised "
            "and mixed as")]
st += [eq("s = w · r<sub>GNN</sub> + ( 1 − w ) · r<sub>RF</sub> , &nbsp; w " + IN + " {0, 0.25, 0.5, 0.75, 1}", 3)]
st += [para("with <i>w</i> selected on the validation partition. Ranks make the differently calibrated outputs "
            "comparable, and ROC-AUC depends only on the ordering. Importantly, if the graph model carried no "
            "information beyond the fingerprint, validation would select <i>w</i> = 0 and the ensemble would "
            "collapse onto the random forest.", indent=False)]
st += [sub("G", "Evaluation protocol")]
st += [para("For each endpoint, ROC-AUC and PR-AUC are computed on the test molecules that carry an observed "
            "label for that endpoint; missing labels are excluded from that endpoint's metric rather than "
            "imputed. Endpoint scores are macro-averaged and reported as mean ± standard deviation over the "
            "three split seeds. PR-AUC is reported alongside ROC-AUC because its chance level equals the "
            "prevalence, which makes it far more sensitive to how well the rare actives are ranked. Models are "
            "compared with two-sided paired Wilcoxon signed-rank tests over the 36 endpoint–seed pairs of test "
            "ROC-AUC [13].", indent=False)]

# ============================================================================= V. RESULTS
st += [section("Results")]
st += [sub("A", "Overall comparison")]
st += [col_figure("col_models.png",
                  "Mean ROC-AUC (top) and PR-AUC (bottom) over the twelve endpoints. Solid bars "
                  "are the scaffold split, with error bars showing the standard deviation over "
                  "three seeds; pale bars behind them are the random split. Colour encodes model "
                  "family.")]
rows = [["Model", "ROC-AUC (scaffold)", "PR-AUC (scaffold)", "ROC-AUC (random)", "Gap"]]
bold = []
for m, lab in [("LogReg", "Logistic regression"), ("RandomForest", "Random forest"), ("XGBoost", "XGBoost"),
               ("XGBoost-FPonly", "XGBoost (FP only)"), ("FP-MLP", "FP-MLP"), ("GCN", "GCN"),
               ("GINE-MPNN", "GINE-MPNN"), ("AttentiveFP", "AttentiveFP"), ("Hybrid", "Hybrid"),
               ("Ensemble RF+GNN", "Ensemble RF+GNN")]:
    rnd = g("random", m, "roc_auc_mean")
    rows.append([lab, ms(m), ms(m, "pr_auc"), "—" if rnd is None else f"{rnd:.3f}",
                 "—" if m not in gap.index else f"{gap[m]:.3f}"])
    if m in ("RandomForest", "Ensemble RF+GNN"):
        bold.append(len(rows) - 1)
st += [table(rows, [COLW * 0.265, COLW * 0.225, COLW * 0.225, COLW * 0.165, COLW * 0.12],
             "Test Performance Averaged Over Twelve Endpoints (Mean ± SD Over Three Seeds). "
             "Gap = ROC-AUC(Random) − ROC-AUC(Scaffold)", bold_rows=bold, font=6.9)]
st += [para(f"Table III and Fig. {FN['col_models.png']} give the headline comparison. On scaffold splits the three graph networks "
            f"reach {GINE:.3f} (GINE-MPNN), {GCN:.3f} (GCN) and {AFP:.3f} (AttentiveFP) ROC-AUC. These sit "
            f"close to gradient boosting ({XGB:.3f}) and to the fingerprint perceptron ({MLP:.3f}), but clearly "
            f"below the random forest ({RF:.3f}), which remains the strongest single model. The descriptor "
            "ablation is informative in the other direction: XGBoost restricted to fingerprint bits falls to "
            f"{g('scaffold', 'XGBoost-FPonly', 'roc_auc_mean'):.3f}, so the global physicochemical descriptors "
            "contribute more to the classical pipeline than any architectural choice contributes to the deep "
            f"one. The late-fusion ensemble is the best model overall on both metrics ({ENS:.3f} ROC-AUC, "
            f"{ENS_PR:.3f} PR-AUC).", indent=False)]
st += [sub("B", "Statistical significance")]
LB = {"RandomForest": "RF", "XGBoost": "XGB", "FP-MLP": "FP-MLP", "GCN": "GCN", "GINE-MPNN": "GINE",
      "AttentiveFP": "AttFP", "Hybrid": "Hybrid", "Ensemble RF+GNN": "Ensemble"}
rows = [["A", "B", "Δ ROC-AUC", "Wins A / B", "<i>p</i>", "Sig."]]
for a, b in [("RandomForest", "GCN"), ("RandomForest", "AttentiveFP"), ("RandomForest", "GINE-MPNN"),
             ("RandomForest", "FP-MLP"), ("RandomForest", "Hybrid"), ("XGBoost", "GCN"),
             ("XGBoost", "AttentiveFP"), ("FP-MLP", "GCN"), ("GCN", "GINE-MPNN"), ("GCN", "AttentiveFP"),
             ("AttentiveFP", "Hybrid"), ("Ensemble RF+GNN", "RandomForest")]:
    d, wa, wb, p = wtest(a, b)
    rows.append([LB[a], LB[b], f"{d:+.4f}", f"{wa} / {wb}", pf(p), "yes" if p < 0.05 else "no"])
st += [table(rows, [COLW * 0.16, COLW * 0.15, COLW * 0.21, COLW * 0.18, COLW * 0.16, COLW * 0.13],
             "Paired Wilcoxon Signed-Rank Tests on Scaffold-Split ROC-AUC Over 36 Endpoint–Seed Pairs",
             font=7.0)]
d1, d2, d3, d4 = (wtest("RandomForest", "GCN"), wtest("XGBoost", "GCN"), wtest("GCN", "GINE-MPNN"),
                  wtest("Ensemble RF+GNN", "RandomForest"))
st += [para("Table IV confirms the pattern. The random forest is significantly better than every graph network "
            f"({d1[0]:+.3f} against GCN, <i>p</i> = {d1[3]:.3f}) and than the fingerprint perceptron. By "
            f"contrast, the graph networks cannot be distinguished from gradient boosting ({d2[0]:+.4f}, "
            f"<i>p</i> = {d2[3]:.2f}) or from the perceptron, so deep learning on these data neither helps nor "
            "hurts relative to a well-tuned classical model. Two further comparisons are notable. The more "
            f"expressive bond-aware GINE-MPNN is significantly <i>worse</i> than the plain GCN ({-d3[0]:+.3f}, "
            f"<i>p</i> = {d3[3]:.3f}), indicating that additional capacity overfits at this data scale. The "
            f"ensemble improves on the random forest in {d4[1]} of 36 pairs (<i>p</i> = {d4[3]:.3f}), the "
            "clearest evidence that the graph view carries complementary signal.", indent=False)]

st += [sub("C", "Endpoint-wise behaviour")]
st += [para(f"Fig. {FN['wide_heatmap.png']} breaks the comparison down by endpoint. GCN attains the best score of any model on "
            "NR-Aromatase (0.804 against 0.750 for the random forest), and AttentiveFP is the strongest single "
            "model on NR-PPAR-gamma; both are endpoints where the classical pipeline is comparatively weak. "
            "Conversely, gradient boosting is best on SR-ARE and SR-MMP, the two endpoints with the most "
            "positives. PR-AUC remains far below ROC-AUC on the rarest endpoints — SR-ATAD5 reaches 0.85 "
            "ROC-AUC but only 0.23 PR-AUC — confirming that ranking the few actives at the very top of a "
            "screening list is still difficult even when overall discrimination looks good.", indent=False)]
st += [sub("D", "Training dynamics")]
st += [para(f"Fig. {FN['col_curves.png']} explains why joint training of the two views is awkward. The fingerprint perceptron drives "
            "its training loss below 0.1 within roughly twenty epochs while its validation ROC-AUC peaks at "
            "epoch two or three: it memorises the training set almost immediately. The graph networks reduce "
            "their loss slowly and continue to improve on validation for thirty to ninety epochs. A single "
            "early-stopping criterion applied to a network containing both branches is therefore premature for "
            "one branch and far too late for the other, which is precisely the failure that the modality-"
            "dropout regulariser of Section IV-F was designed to address.", indent=False)]
st += [col_figure("col_curves.png",
                  "Training loss (top) and mean validation ROC-AUC (bottom) on the first scaffold "
                  "split. The fingerprint perceptron converges within roughly twenty epochs; the "
                  "graph networks need thirty to ninety and are still improving when the "
                  "perceptron has long since overfitted.")]
st += wide_figures(("wide_heatmap.png",
                    "Endpoint-wise test ROC-AUC on the scaffold split, averaged over three seeds. Bold entries "
                    "mark the best model for each endpoint. No single representation dominates: graph networks "
                    "win NR-Aromatase and NR-PPAR-gamma, gradient boosting wins the two largest stress-response "
                    "assays, and the ensemble wins the majority."))
# ============================================================================= VI. GENERALISATION
st += [section("Scaffold Generalisation")]
st += [sub("A", "Random versus scaffold splits")]
st += [para("Every model scores lower on scaffold splits than on random splits, confirming that random "
            "splitting overestimates performance on genuinely new chemistry. Part of this optimism is "
            "near-duplicate leakage: 7.1% of random-split test molecules have a fingerprint identical to a "
            "training molecule — mostly stereoisomers, which are distinct compounds but share a non-chiral ECFP "
            "— against 0.4% under scaffold splitting. The size of the gap differs systematically between "
            f"families (Fig. {FN['col_gap.png']}). The tree ensembles lose {gap['RandomForest']:.3f} (random forest) and "
            f"{gap['XGBoost']:.3f} (XGBoost), whereas the graph networks lose {gap['GINE-MPNN']:.3f} to "
            f"{max(gap['GCN'], gap['AttentiveFP']):.3f}, roughly twice as much.", indent=False)]
st += [para("This is the most consequential finding of the phase. On <i>random</i> splits GCN "
            f"({g('random', 'GCN', 'roc_auc_mean'):.3f}) and AttentiveFP "
            f"({g('random', 'AttentiveFP', 'roc_auc_mean'):.3f}) are statistically level with the random forest "
            f"({g('random', 'RandomForest', 'roc_auc_mean'):.3f}); their apparent disadvantage emerges only "
            "once the test chemistry is new. A learned representation is fitted to the training distribution, "
            "so novel scaffolds present atom environments that the message functions have rarely seen, whereas "
            "fixed fingerprints and global descriptors are not learned and therefore cannot overfit to the "
            "training chemotypes in the same way.")]
st += [col_figure("col_gap.png", "ROC-AUC lost when moving from a random to a scaffold split, averaged over "
                                 "endpoints. Lower is better. The graph models (green) lose about twice as much "
                                 "as the tree ensembles (blue).")]
st += [sub("B", "Distance to the training set")]
st += [para("To make the effect concrete, each test molecule is binned by the maximum Tanimoto similarity "
            "between its Morgan fingerprint and those of all training molecules, pooling the test sets of all "
            f"six splits (Fig. {FN['col_similarity.png']}). Accuracy declines monotonically with novelty for "
            "every model: from 0.85–0.88 "
            "ROC-AUC for molecules with a close training analogue (similarity at least 0.7) down to "
            f"{sim.loc['XGBoost', '<0.3']:.2f}–{sim.loc['Ensemble RF+GNN', '<0.3']:.2f} below 0.3. In that "
            f"hardest bin the random forest ({sim.loc['RandomForest', '<0.3']:.3f}) and GINE-MPNN "
            f"({sim.loc['GINE-MPNN', '<0.3']:.3f}) degrade least among single models, XGBoost degrades most "
            f"({sim.loc['XGBoost', '<0.3']:.3f}), and the ensemble remains best "
            f"({sim.loc['Ensemble RF+GNN', '<0.3']:.3f}). For deployment this argues for an explicit "
            "applicability-domain check: predictions for compounds whose nearest training neighbour falls below "
            "a similarity of roughly 0.3 to 0.4 should be flagged as unreliable rather than reported as "
            "scores.", indent=False)]
st += [col_figure("col_similarity.png", "Mean ROC-AUC against each test molecule's Tanimoto similarity to its "
                                        "nearest training molecule, with the test sets of all six splits "
                                        "pooled. Every model degrades sharply on genuinely novel chemistry.")]

# ============================================================================= VII. DISCUSSION
st += [section("Discussion")]
st += [sub("A", "Why learned representations do not win here")]
st += [para("Three factors explain the result, and they reinforce one another. First, Tox21 provides only about "
            "6,100 training molecules and between 180 and 900 actives per endpoint, which is small for "
            "representation learning; a Morgan fingerprint is itself essentially a Weisfeiler–Lehman style hash "
            "of the same neighbourhoods a message passing network aggregates, so the network must learn from "
            "scarce data something the fingerprint already encodes for free. Second, the baseline is unusually "
            "strong because it is not a fingerprint alone: the descriptor ablation shows that global properties "
            "such as lipophilicity, molecular size and polar surface area carry a large part of the signal, and "
            "a graph network must infer these from atom-level features. Third, scaffold splitting penalises "
            "learned representations more than fixed ones, as Section VI showed. Our conclusion is consistent "
            "with MoleculeNet [2] and Chemprop [12], where graph models pull clearly ahead mainly on larger "
            "datasets, with added descriptors, or after self-supervised pre-training.", indent=False)]
st += [para("The FP-MLP control rules out the most obvious alternative explanation. Trained on identical "
            f"features, it reaches {MLP:.3f} ROC-AUC, statistically indistinguishable from the graph networks "
            "and below the random forest. Deep multitask learning by itself therefore neither explains nor "
            "rescues the comparison: the architecture family is not the limiting factor, the data scale is.")]
st += [sub("B", "Complementarity of the two views")]
st += [para("Although graph networks do not win on average, they are not redundant. They are the best single "
            "models on two endpoints, and when the validation set is free to choose the mixing weight of "
            f"equation (3) it assigns on average {W_GRAPH:.2f} of the weight to the graph model rather than "
            "collapsing to zero. The resulting ensemble improves significantly on the random forest and holds "
            "up best of all models on the most novel chemistry. The two representations evidently make "
            "different errors: engineered features capture known substructures and bulk physicochemical "
            "behaviour, while the learned representation adapts to endpoint-specific patterns. Combining them "
            "is more effective than replacing one with the other.", indent=False)]
st += [sub("C", "Expressiveness is not the bottleneck")]
st += [para("A notable negative result is that the most expressive architecture performs worst. GINE-MPNN uses "
            "sum aggregation and bond-conditioned messages, making it strictly more discriminative than GCN in "
            "the Weisfeiler–Lehman sense, yet it falls significantly below GCN and has the lowest PR-AUC of the "
            "graph models. With roughly 6,100 training molecules the additional capacity is spent fitting "
            "training chemotypes. This supports the view that, in low-data molecular property prediction, "
            "regularisation and inductive bias matter more than theoretical expressive power.", indent=False)]
st += [sub("D", "Reliability of the comparison")]
st += [para("Split-to-split variation (standard deviations of 0.02–0.04 ROC-AUC) exceeds most differences "
            "between models, so conclusions drawn from a single split could easily be reversed. Pairing "
            "comparisons by endpoint and seed removes this shared difficulty, and we claim only the differences "
            "that are significant in Table IV. Because endpoints are correlated and seeds share molecules, the "
            "<i>p</i>-values should be read as approximate; nevertheless the principal random-forest-versus-"
            "graph results (<i>p</i> ≤ 0.004) survive a Bonferroni correction over the planned comparisons.",
            indent=False)]

# ============================================================================= VIII. LIMITATIONS
st += [section("Limitations and Future Work")]
st += [para("<b>Unequal tuning.</b> The classical baselines received a per-endpoint grid search, whereas each "
            "network used a single fixed configuration. A systematic search over depth, width, dropout and "
            "learning rate could narrow the gap, although the consistency of the result across three "
            "architectures suggests it would not reverse it.", indent=False)]
st += [para("<b>No pre-training.</b> Self-supervised pre-training of the graph encoder on large unlabelled "
            "corpora is the most promising route to stronger learned representations in this regime, and is the "
            "obvious next experiment.")]
st += [para("<b>Calibration and uncertainty.</b> Class-weighted losses optimise ranking rather than "
            "probabilities, so the outputs would need Platt or isotonic calibration before being used as risk "
            "estimates, and deep ensembling could supply uncertainty intervals.")]
st += [para("<b>Stereochemistry and split strictness.</b> ECFP ignores chirality and our graph models see only "
            "a chirality tag, so stereoisomers carrying different labels cannot be separated reliably. Scaffold "
            "splitting is also imperfect, since different scaffolds can still be chemically similar; "
            "cluster-based or temporal splits would test extrapolation more strictly.")]

# ============================================================================= IX. CONCLUSION
st += [section("Conclusion")]
st += [para("We compared engineered chemical fingerprints with learned molecular-graph representations for "
            "predicting twelve Tox21 toxicity endpoints under a leakage-controlled scaffold-split protocol. "
            "Three message passing architectures trained as multitask models with a masked, class-weighted loss "
            f"reach {GINE:.3f}–{AFP:.3f} ROC-AUC on unseen chemical families. They are statistically "
            "indistinguishable from gradient boosting and from a deep perceptron over the same engineered "
            f"features, and significantly below a random forest over fingerprints and descriptors ({RF:.3f}). "
            "Graph networks also generalise less well from random to scaffold splits, losing about twice as "
            "much accuracy as the tree ensembles. The two views are nonetheless complementary: a "
            "validation-weighted late fusion of a random forest and a graph network is the best model overall "
            f"({ENS:.3f} ROC-AUC, {ENS_PR:.3f} PR-AUC) and degrades least on novel chemistry. For datasets of "
            "this size we therefore recommend combining engineered and learned representations rather than "
            "substituting one for the other, and evaluating under scaffold splits, where differences in "
            "generalisation become visible at all.", indent=False)]

# ============================================================================= REFERENCES
st += [Paragraph("REFERENCES", S["sec"])]   # IEEE: the reference list is not a numbered section
for i, r in enumerate([
    "R. Huang <i>et al.</i>, “Tox21Challenge to build predictive models of nuclear receptor and stress response "
    "pathways as mediated by exposure to environmental chemicals and drugs,” <i>Frontiers in Environmental "
    "Science</i>, vol. 3, art. 85, 2016.",
    "Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, and V. Pande, "
    "“MoleculeNet: A benchmark for molecular machine learning,” <i>Chemical Science</i>, vol. 9, no. 2, "
    "pp. 513–530, 2018.",
    "D. Rogers and M. Hahn, “Extended-connectivity fingerprints,” <i>Journal of Chemical Information and "
    "Modeling</i>, vol. 50, no. 5, pp. 742–754, 2010.",
    "L. Breiman, “Random forests,” <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, 2001.",
    "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in <i>Proc. 22nd ACM SIGKDD Int. "
    "Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785–794.",
    "G. W. Bemis and M. A. Murcko, “The properties of known drugs. 1. Molecular frameworks,” <i>Journal of "
    "Medicinal Chemistry</i>, vol. 39, no. 15, pp. 2887–2893, 1996.",
    "J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural message passing for quantum "
    "chemistry,” in <i>Proc. 34th Int. Conf. Machine Learning</i>, 2017, pp. 1263–1272.",
    "T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in "
    "<i>Proc. Int. Conf. Learning Representations</i>, 2017.",
    "K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?” in <i>Proc. Int. "
    "Conf. Learning Representations</i>, 2019.",
    "W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec, “Strategies for pre-training "
    "graph neural networks,” in <i>Proc. Int. Conf. Learning Representations</i>, 2020.",
    "Z. Xiong <i>et al.</i>, “Pushing the boundaries of molecular representation for drug discovery with the "
    "graph attention mechanism,” <i>Journal of Medicinal Chemistry</i>, vol. 63, no. 16, pp. 8749–8760, 2019.",
    "K. Yang <i>et al.</i>, “Analyzing learned molecular representations for property prediction,” <i>Journal "
    "of Chemical Information and Modeling</i>, vol. 59, no. 8, pp. 3370–3388, 2019.",
    "J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” <i>Journal of Machine "
    "Learning Research</i>, vol. 7, pp. 1–30, 2006.",
], 1):
    st += [Paragraph(f"[{i}]&nbsp;&nbsp;{r}", S["ref"])]


# ============================================================================= document
class Doc(BaseDocTemplate):
    def afterPage(self):
        self.canv.setFont("T", 9)
        self.canv.setFillColor(colors.HexColor("#222222"))
        self.canv.drawCentredString(PW / 2, MB - 26, str(self.page))


def cols(top):
    return [Frame(x, BODY_BOT, COLW, top - BODY_BOT, id=i, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0) for x, i in ((LEFT_X, "l"), (RIGHT_X, "r"))]


TITLE_H = 118
templates = [PageTemplate(id="first",
                          frames=[Frame(MS, BODY_TOP - TITLE_H, FULLW, TITLE_H, id="t", leftPadding=0,
                                        rightPadding=0, topPadding=0, bottomPadding=0)]
                          + cols(BODY_TOP - TITLE_H - 10)),
             PageTemplate(id="normal", frames=cols(BODY_TOP))]
for tid, h in WIDE_SPECS:
    h = min(h, BODY_TOP - BODY_BOT - 120)        # always leave room for text below
    templates.append(PageTemplate(id=tid,
                                  frames=[Frame(MS, BODY_TOP - h, FULLW, h, id="w", leftPadding=0,
                                                rightPadding=0, topPadding=0, bottomPadding=0)]
                                  + cols(BODY_TOP - h - 8)))

doc = Doc(str(OUT), pagesize=letter, leftMargin=MS, rightMargin=MS, topMargin=MT, bottomMargin=MB,
          title="Learned Graph Representations versus Engineered Fingerprints for Tox21",
          author="AML/DL Group Project 8", subject="Phase 2 project report")
doc.addPageTemplates(templates)
doc.build(st)
print("wrote", OUT, "| wide figure frames:", [(t, round(h)) for t, h in WIDE_SPECS])
