#!/usr/bin/env python3
"""Generate docs/HPL_Pipeline_Architecture.pdf.

The PDF is a build artifact; this script is the source. Regenerate after
changing the pipeline:

    python docs/make_architecture_pdf.py

Needs reportlab (`pip install reportlab`) and nothing else — no LaTeX, no
pandoc, no network. Everything it describes is stated in one place here so the
document cannot drift section by section.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

OUT = Path(__file__).resolve().parent / "HPL_Pipeline_Architecture.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5c5c5c")
RULE = colors.HexColor("#d4d4d4")
ACCENT = colors.HexColor("#1f4e79")
WARN = colors.HexColor("#9c2b0e")
OK = colors.HexColor("#1e6b3a")
BAND = colors.HexColor("#f2f4f7")
HPC = colors.HexColor("#e8eef5")
LOCAL = colors.HexColor("#eef5ee")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------

_base = getSampleStyleSheet()

S = {
    "title": ParagraphStyle(
        "title", parent=_base["Title"], fontName="Helvetica-Bold",
        fontSize=26, leading=30, textColor=INK, alignment=TA_LEFT, spaceAfter=4),
    "subtitle": ParagraphStyle(
        "subtitle", parent=_base["Normal"], fontName="Helvetica",
        fontSize=12.5, leading=17, textColor=MUTED, alignment=TA_LEFT),
    "h1": ParagraphStyle(
        "h1", parent=_base["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=ACCENT, spaceBefore=2, spaceAfter=7),
    "h2": ParagraphStyle(
        "h2", parent=_base["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, textColor=INK, spaceBefore=11, spaceAfter=4),
    "body": ParagraphStyle(
        "body", parent=_base["Normal"], fontName="Helvetica",
        fontSize=9.4, leading=13.4, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle(
        "small", parent=_base["Normal"], fontName="Helvetica",
        fontSize=8.2, leading=11.4, textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle(
        "cell", parent=_base["Normal"], fontName="Helvetica",
        fontSize=8.1, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle(
        "cellb", parent=_base["Normal"], fontName="Helvetica-Bold",
        fontSize=8.1, leading=10.8, textColor=INK),
    "cellhead": ParagraphStyle(
        "cellhead", parent=_base["Normal"], fontName="Helvetica-Bold",
        fontSize=8.1, leading=10.8, textColor=colors.white),
    "code": ParagraphStyle(
        "code", parent=_base["Normal"], fontName="Courier",
        fontSize=8.0, leading=11.2, textColor=INK,
        backColor=BAND, borderPadding=5, spaceBefore=3, spaceAfter=7),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(text):
    return Paragraph(text, S["h1"])


def H2(text):
    return Paragraph(text, S["h2"])


def code(text):
    return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])


def note(text, colour=WARN):
    """A one-line callout with a coloured left rule."""
    t = Table([[Paragraph(text, S["body"])]], colWidths=[CONTENT_W - 6])
    t.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, colour),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
    ]))
    return KeepTogether([Spacer(1, 2), t, Spacer(1, 7)])


def table(rows, widths, head=True, size=8.1):
    data = []
    for i, row in enumerate(rows):
        style = "cellhead" if (head and i == 0) else "cell"
        data.append([c if isinstance(c, Flowable) else Paragraph(str(c), S[style])
                     for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]
    if head:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
        for r in range(2, len(data), 2):
            cmds.append(("BACKGROUND", (0, r), (-1, r), BAND))
    t.setStyle(TableStyle(cmds))
    return t


# --------------------------------------------------------------------------
# diagrams
# --------------------------------------------------------------------------

class Diagram(Flowable):
    """Base for the hand-drawn figures. Subclasses implement paint()."""

    def __init__(self, width, height):
        super().__init__()
        self.width, self.height = width, height

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        self.paint(self.canv)

    # -- primitives --------------------------------------------------------

    def box(self, c, x, y, w, h, label, sub=None, fill=colors.white,
            stroke=RULE, bold=True, size=8.4, radius=3):
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(0.8)
        c.roundRect(x, y, w, h, radius, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        if sub:
            c.drawCentredString(x + w / 2, y + h / 2 + 2.5, label)
            c.setFont("Helvetica", size - 1.3)
            c.setFillColor(MUTED)
            c.drawCentredString(x + w / 2, y + h / 2 - 7.5, sub)
        else:
            c.drawCentredString(x + w / 2, y + h / 2 - 3, label)

    def arrow(self, c, x1, y1, x2, y2, colour=MUTED, dash=None, label=None):
        c.setStrokeColor(colour)
        c.setLineWidth(1.0)
        if dash:
            c.setDash(dash, 2)
        c.line(x1, y1, x2, y2)
        c.setDash()
        # head
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 4.2
        c.setFillColor(colour)
        p = c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(x2 - size * math.cos(angle - 0.42), y2 - size * math.sin(angle - 0.42))
        p.lineTo(x2 - size * math.cos(angle + 0.42), y2 - size * math.sin(angle + 0.42))
        p.close()
        c.drawPath(p, stroke=0, fill=1)
        if label:
            c.setFont("Helvetica", 6.8)
            c.setFillColor(MUTED)
            c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 4, label)

    def caption(self, c, x, y, text, size=7.0, colour=MUTED, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColor(colour)
        c.drawString(x, y, text)

    def zone(self, c, x, y, w, h, title, fill):
        c.setFillColor(fill)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.7)
        c.setDash(3, 2)
        c.roundRect(x, y, w, h, 4, stroke=1, fill=1)
        c.setDash()
        c.setFont("Helvetica-Bold", 7.4)
        c.setFillColor(ACCENT)
        c.drawString(x + 7, y + h - 11, title)


class Topology(Diagram):
    """Where each process actually runs, and what the VPN gates."""

    def __init__(self, width):
        super().__init__(width, 168)

    def paint(self, c):
        W = self.width
        left_w = W * 0.30
        right_x = W * 0.42
        right_w = W - right_x

        self.zone(c, 0, 6, left_w, 150, "VAIDEHI'S LAPTOP", LOCAL)
        self.zone(c, right_x, 6, right_w, 150, "BEATSON HPC  (hpc-login-01)", HPC)

        self.box(c, 12, 104, left_w - 24, 30, "Streamlit UI",
                 "app/app_v28.py", fill=colors.white)
        self.box(c, 12, 62, left_w - 24, 30, "React UI (optional)",
                 "frontend/  :5173", fill=colors.white)
        self.box(c, 12, 20, left_w - 24, 30, "api_client.py",
                 "-> http://localhost:8000", fill=colors.white)

        bx = right_x + 14
        bw = right_w - 28
        self.box(c, bx, 112, bw, 30, "FastAPI  tile_server_v2_.py",
                 "port 8000  ·  runs HERE, not on the laptop", fill=colors.white,
                 stroke=ACCENT)
        self.box(c, bx, 70, bw * 0.47, 30, "Slurm",
                 "sbatch / squeue / sacct", fill=colors.white)
        self.box(c, bx + bw * 0.53, 70, bw * 0.47, 30, "Postgres  hpl_kb",
                 "unix socket, 17 tables", fill=colors.white)
        self.box(c, bx, 22, bw, 32, "Scratch filesystem",
                 ".svs slides  ·  tiles  ·  .h5  ·  projections  ·  reference .npz",
                 fill=colors.white)

        # tunnel
        self.arrow(c, left_w - 12 + 24, 35, right_x + 8, 120, ACCENT, dash=(3, 2))
        c.setFont("Helvetica-Bold", 7.0)
        c.setFillColor(ACCENT)
        c.drawCentredString((left_w + right_x) / 2, 92, "VPN +")
        c.drawCentredString((left_w + right_x) / 2, 83, "SSH tunnel")

        self.arrow(c, bx + bw / 2, 112, bx + bw * 0.23, 100)
        self.arrow(c, bx + bw / 2, 112, bx + bw * 0.77, 100)
        self.arrow(c, bx + bw / 2, 70, bx + bw / 2, 54)

        self.caption(c, 0, 0,
                     "The API server runs on the HPC because it shells out to sbatch and "
                     "opens .svs files from scratch. The laptop runs only the UI.")


class PipelineFlow(Diagram):
    """The six stages, their artifacts, and where state is recorded."""

    def __init__(self, width):
        super().__init__(width, 300)

    def paint(self, c):
        W = self.width
        stages = [
            ("1  Tiling", "submit_mask_tile_slurm.py", "tiles + _tile_metadata.csv", "Slurm array"),
            ("2  Packaging", "make_hpl_hdf5.py", "one gzip .h5 per dataset", "Slurm"),
            ("3  Feature extraction", "submit_feature_extraction.py", "projections .h5", "Slurm GPU"),
            ("4  Classification", "submit_cluster_assignment.py", "assignments .csv", "Slurm CPU"),
            ("5  Registration", "register_dataset.py", "identity rows in the KB", "in-process"),
            ("6  KB load", "load_hpc_assignments.py", "hpc_id + aggregates", "in-process"),
        ]
        h = 34
        gap = 10
        top = self.height - 20
        bw = W * 0.60

        for i, (name, module, out, where) in enumerate(stages):
            y = top - (i + 1) * (h + gap)
            fill = colors.white if i < 4 else HPC
            stroke = ACCENT if i >= 4 else RULE
            c.setFillColor(fill)
            c.setStrokeColor(stroke)
            c.setLineWidth(1.0 if i >= 4 else 0.8)
            c.roundRect(0, y, bw, h, 3, stroke=1, fill=1)

            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 9.0)
            c.drawString(9, y + h - 13, name)
            c.setFont("Courier", 7.2)
            c.setFillColor(MUTED)
            c.drawString(9, y + 8, module)

            c.setFont("Helvetica", 7.4)
            c.setFillColor(INK)
            c.drawString(bw + 14, y + h - 13, out)
            c.setFont("Helvetica", 6.8)
            c.setFillColor(MUTED)
            c.drawString(bw + 14, y + 8, where)

            if i:
                self.arrow(c, bw / 2, y + h + gap, bw / 2, y + h + 1.5)

        y5 = top - 5 * (h + gap)
        y6 = top - 6 * (h + gap)
        c.setStrokeColor(WARN)
        c.setLineWidth(1.0)
        c.setDash(2, 2)
        c.line(bw + 6, y5 + h / 2, bw + 6, y6 + h / 2)
        c.setDash()
        c.setFont("Helvetica-Bold", 6.6)
        c.setFillColor(WARN)
        c.drawString(bw + 10, (y5 + y6) / 2 + h / 2 - 2, "6 refuses at 0% match without 5")

        self.caption(c, 0, 2,
                     "Stages 1-4 submit Slurm jobs. Stages 5 and 6 run inside the API "
                     "process and commit in one transaction each.")


class KbMap(Diagram):
    """The 17 tables, grouped by what fills them."""

    def __init__(self, width):
        super().__init__(width, 236)

    def paint(self, c):
        W = self.width
        col_w = (W - 20) / 3
        groups = [
            ("FILLED BY A RUN", OK, [
                ("tile_coordinates", "Stage 5"),
                ("tile_registry", "Stage 5 + 6"),
                ("wsi_registry", "Stage 5"),
                ("wsi_metadata", "Stage 5 (opt-in)"),
                ("dataset_config", "Stage 5"),
                ("hpl_profile_summary", "Stage 6"),
                ("hpl_profile_proportion", "Stage 6"),
                ("slide_hpc_membership", "Stage 6"),
                ("slurm_dataset_runs", "the server"),
                ("slurm_dataset_run_jobs", "the server"),
            ]),
            ("STATIC REFERENCE", ACCENT, [
                ("hpc_dictionary", "71 clusters"),
                ("hpc_malignant_details", "27 rows"),
                ("hpc_non_malignant_details", "44 rows"),
                ("hpc_survival_analysis", "Cox coefs"),
            ]),
            ("NOTHING FILLS THESE", WARN, [
                ("tile_hpc_heatmap", "READ LIVE - 149 MB"),
                ("h_latent_vectors", "no reader - 4.4 GB"),
                ("tile_hpc_heatmap_old", "legacy - 141 MB"),
                ("slide_metadata (view)", "no reader"),
            ]),
        ]

        for gi, (title, colour, items) in enumerate(groups):
            x = gi * (col_w + 10)
            c.setFillColor(colour)
            c.rect(x, self.height - 16, col_w, 14, stroke=0, fill=1)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 7.2)
            c.drawString(x + 6, self.height - 12, title)

            for ii, (name, sub) in enumerate(items):
                y = self.height - 36 - ii * 22
                c.setFillColor(colors.white)
                c.setStrokeColor(RULE)
                c.setLineWidth(0.6)
                c.rect(x, y, col_w, 19, stroke=1, fill=1)
                c.setFillColor(colour)
                c.rect(x, y, 2.2, 19, stroke=0, fill=1)
                c.setFillColor(INK)
                c.setFont("Helvetica-Bold", 7.0)
                c.drawString(x + 7, y + 11, name)
                c.setFont("Helvetica", 6.3)
                c.setFillColor(MUTED)
                c.drawString(x + 7, y + 3, sub)

        self.caption(c, 0, 0,
                     "tile_hpc_heatmap is the one open gap with a live reader: the server "
                     "loads it at startup and nothing has ever written it.")


class ClassifierFlow(Diagram):
    """How a tile becomes a cluster id."""

    def __init__(self, width):
        super().__init__(width, 150)

    def paint(self, c):
        W = self.width
        bw = (W - 4 * 11) / 5
        y = 76
        steps = [
            ("tile", "224px @ 5x"),
            ("encoder", "frozen SSL -> 128-D"),
            ("project", "PCA basis + centering"),
            ("faiss", "IndexFlatL2, exact"),
            ("vote", "k=10, 1/d^3"),
        ]
        for i, (name, sub) in enumerate(steps):
            x = i * (bw + 11)
            self.box(c, x, y, bw, 36, name, sub,
                     fill=colors.white, stroke=ACCENT if i in (3, 4) else RULE)
            if i:
                self.arrow(c, x - 10, y + 18, x - 1.5, y + 18)

        # The reference feeds the search, and the vote produces the row. Drawn
        # that way round: an arrow from the reference to the output would say
        # the labels come straight off the reference file.
        out_w = W * 0.42
        ref_x = W * 0.56
        self.box(c, 0, 22, out_w, 30, "hpc_id + vote_margin",
                 "-> tile_registry", fill=HPC, stroke=ACCENT)
        self.box(c, ref_x, 22, W - ref_x, 30, "reference .npz",
                 "2.5M LATTICeA tiles, 71 clusters", fill=BAND)

        faiss_cx = 3 * (bw + 11) + bw / 2
        vote_cx = 4 * (bw + 11) + bw / 2
        # reference -> faiss
        self.arrow(c, ref_x + (W - ref_x) / 2, 52, faiss_cx, y - 1.5, ACCENT)
        # vote -> the registry row
        self.arrow(c, vote_cx, y - 1.5, out_w / 2, 53, ACCENT)

        self.caption(c, 0, 4,
                     "Re-voted at k=25 when vote_margin < 0.15. Closed-book accuracy on "
                     "unseen slides: 97.33%.")


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------

def _decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, MARGIN - 5, PAGE_W - MARGIN, MARGIN - 5)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, MARGIN - 14,
                      "HPL HPC assignment pipeline — technical architecture")
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 14, str(canvas.getPageNumber()))
    canvas.restoreState()


def build(story):
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN + 6,
        title="HPL HPC Assignment Pipeline — Technical Architecture",
        author="Generated by docs/make_architecture_pdf.py",
        subject="Pipeline, Knowledge Bank and deployment architecture",
    )
    frame = Frame(MARGIN, MARGIN + 6, CONTENT_W,
                  PAGE_H - 2 * MARGIN - 6, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_decorate)])
    doc.build(story)


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------

def story():
    s = []

    # ---- cover -----------------------------------------------------------
    s += [
        Spacer(1, 34),
        P("HPL HPC Assignment Pipeline", "title"),
        Spacer(1, 3),
        P("Technical architecture — pipeline stages, the Knowledge Bank, "
          "and how the pieces are deployed", "subtitle"),
        Spacer(1, 12),
    ]
    s.append(table([
        ["Version", "2026-08-26"],
        ["Scope", "backend/ (FastAPI + Slurm submitters), app/ (Streamlit UI), "
                  "frontend/ (React UI), Postgres <b>hpl_kb</b>"],
        ["Cluster", "Beatson HPC, University of Glasgow — Slurm, Singularity, "
                    "NVIDIA H200/H100/A100"],
        ["Current UI", "app/app_v28.py &nbsp;·&nbsp; current server: "
                       "backend/tile_server_v2_.py"],
        ["Reference", "hpc_reference_leiden_2p5_fold2.npz — LATTICeA, leiden 2.5, "
                      "fold 2, 2.5M tiles, 71 clusters"],
        ["Source", "docs/make_architecture_pdf.py — regenerate rather than edit"],
    ], [CONTENT_W * 0.20, CONTENT_W * 0.80], head=False))

    s += [
        Spacer(1, 16),
        H1("What this system does"),
        P("Whole-slide H&amp;E images are cut into tiles, each tile is turned into a "
          "128-dimensional vector by a frozen self-supervised encoder, and each vector "
          "is assigned to one of 71 <b>histomorphological phenotype clusters</b> (HPCs) "
          "by exact k-nearest-neighbour search against a reference built from the "
          "LATTICeA cohort. The results land in a Postgres <b>Knowledge Bank</b> that a "
          "slide viewer, an HPC explorer and a natural-language chatbot read."),
        P("The pipeline is six stages. The first four submit Slurm jobs and produce "
          "files; the last two write to the Knowledge Bank from inside the API process. "
          "Every stage is gated in the UI on the previous stage having produced a "
          "<i>valid</i> output — not merely on it having run."),
    ]

    s += [Spacer(1, 8), H2("Deployment — what runs where"), Topology(CONTENT_W)]
    s += [
        note("The API server runs <b>on the HPC login node</b>, not on the laptop. It "
             "shells out to <font face='Courier'>sbatch</font>, "
             "<font face='Courier'>squeue</font> and "
             "<font face='Courier'>sacct</font>, opens <font face='Courier'>.svs</font> "
             "files directly from scratch, and reaches Postgres over a local socket. "
             "The laptop runs the UI only, and reaches the API through the university "
             "VPN plus an SSH tunnel to port 8000. Deploying a change to "
             "<font face='Courier'>backend/</font> therefore means copying it to the "
             "cluster and restarting the server — running it locally is not enough.",
             WARN),
        P("Two access paths exist, and the split matters when planning a change. Most of "
          "the UI goes through <font face='Courier'>api_client.TileServerClient</font> "
          "over HTTP. The chatbot and three viewer helpers "
          "(<font face='Courier'>load_hpc_titles</font>, "
          "<font face='Courier'>load_survival_coefficients</font>, "
          "<font face='Courier'>slides_for_hpc_from_kb</font>) query Postgres "
          "<i>directly</i> via SQLAlchemy, bypassing the API. Those need a second tunnel "
          "to 5432. The React frontend has no direct database path at all — it is "
          "HTTP-only, which is why the chatbot was deliberately left out of that port."),
    ]

    # ---- pipeline --------------------------------------------------------
    s += [PageBreak(), H1("The pipeline, stage by stage"), PipelineFlow(CONTENT_W)]

    s.append(table([
        ["Stage", "Runs where", "Reads", "Writes", "State tracked"],
        ["<b>1 Tiling</b><br/><font face='Courier' size='7'>submit_mask_<br/>tile_slurm.py</font>",
         "Slurm array,<br/>1 CPU/task",
         "raw <font face='Courier'>.svs</font>",
         "tiles on disk +<br/>per-slide <font face='Courier'>_tile_metadata.csv</font>",
         "<font face='Courier'>job_id</font>, manifest;<br/>completion recomputed"],
        ["<b>2 Packaging</b><br/><font face='Courier' size='7'>make_hpl_hdf5.py</font>",
         "Slurm,<br/>8 CPUs",
         "tiles + CSVs",
         "one gzip <font face='Courier'>.h5</font> per dataset",
         "<font face='Courier'>h5_job_id</font>,<br/><font face='Courier'>h5_output_path</font>"],
        ["<b>3 Feature extraction</b><br/><font face='Courier' size='7'>submit_feature_<br/>extraction.py</font>",
         "Slurm GPU,<br/>Singularity",
         "packaged <font face='Courier'>.h5</font>",
         "projections <font face='Courier'>.h5</font><br/>(<font face='Courier'>z_latent</font>, <font face='Courier'>h_latent</font>)",
         "<font face='Courier'>extraction_job_id</font>,<br/>checkpoint"],
        ["<b>4 Classification</b><br/><font face='Courier' size='7'>submit_cluster_<br/>assignment.py</font>",
         "Slurm CPU,<br/>faiss",
         "projections + reference <font face='Courier'>.npz</font>",
         "assignments <font face='Courier'>.csv</font>",
         "<font face='Courier'>assignment_job_id</font>,<br/><font face='Courier'>assignment_vote</font>"],
        ["<b>5 Registration</b><br/><font face='Courier' size='7'>register_dataset.py</font>",
         "in the API<br/>process",
         "raw slides, Stage 1 CSVs, packaged <font face='Courier'>.h5</font>",
         "<font face='Courier'>wsi_registry</font>, <font face='Courier'>wsi_metadata</font>, <font face='Courier'>dataset_config</font>, <font face='Courier'>tile_coordinates</font>, <font face='Courier'>tile_registry</font>",
         "<font face='Courier'>registration_done</font>"],
        ["<b>6 KB load</b><br/><font face='Courier' size='7'>load_hpc_assignments.py</font>",
         "in the API<br/>process",
         "assignments <font face='Courier'>.csv</font>",
         "<font face='Courier'>tile_registry.hpc_id</font>,<br/><font face='Courier'>hpl_profile_*</font>",
         "<font face='Courier'>kb_load_done</font>"],
    ], [CONTENT_W * 0.22, CONTENT_W * 0.12, CONTENT_W * 0.17,
        CONTENT_W * 0.29, CONTENT_W * 0.20]))

    s += [
        Spacer(1, 9),
        note("<b>Stage 1 does not write <font face='Courier'>tile_coordinates</font>.</b> "
             "It writes a per-slide CSV to disk. Nothing puts those rows into Postgres "
             "until Stage 5. The project documentation asserted otherwise until "
             "2026-08-26, and that mistake is what kept the registration gap invisible: "
             "everyone believed the coordinates table filled itself.", WARN),
        H2("Why Stage 5 exists"),
        P("<font face='Courier'>load_hpc_assignments.load()</font> only ever "
          "<font face='Courier'>UPDATE</font>s "
          "<font face='Courier'>tile_registry.hpc_id</font> — it never creates a row. For "
          "a cohort that has never touched the Knowledge Bank there is nothing to update, "
          "so Stage 6's match rate is 0% <i>by construction</i> and it refuses. The "
          "refusal is correct and its message names a match rate, which reads like a "
          "slide-naming bug. It is in fact a missing step."),
        P("Stage 5 creates the identity rows Stage 6 needs. It is gated on Stage 2, not "
          "Stage 4: it reads tile identity out of the packaged "
          "<font face='Courier'>.h5</font> and the raw slides and needs no cluster labels, "
          "so it can finish while the GPU work is still queued."),
        P("All five tables are written in <b>one transaction</b>. A registration that "
          "wrote <font face='Courier'>tile_registry</font> but not "
          "<font face='Courier'>wsi_registry</font> would pass every check the pipeline "
          "has — Stage 6 loads, the aggregates refresh, every table grows — and the "
          "cohort would still be invisible in the viewer, because "
          "<font face='Courier'>_open_slide()</font> resolves slide paths from "
          "<font face='Courier'>wsi_registry</font> alone."),
    ]

    # ---- knowledge bank --------------------------------------------------
    # No PageBreak here: the two paragraphs above spill onto the next page,
    # and forcing a break left that page four-fifths empty.
    s += [Spacer(1, 10), H1("The Knowledge Bank"), P(
        "Postgres <font face='Courier'>hpl_kb</font>: 17 tables, one view, 45 indexes. "
        "Column-level definitions live in "
        "<font face='Courier'>backend/kb_live_schema_2026-08-26.txt</font>, transcribed "
        "from <font face='Courier'>\\d</font> against the live database."),
        KbMap(CONTENT_W)]

    s += [
        Spacer(1, 8),
        H2("The join key"),
        P("<font face='Courier'>slide_tile</font> is "
          "<font face='Courier'>\"&lt;slides&gt;_&lt;tiles&gt;\"</font> upper-cased — "
          "<font face='Courier'>TCGA-55-7574-01Z-00-DX1_18_15.JPEG</font>. It is the "
          "primary key of both <font face='Courier'>tile_coordinates</font> and "
          "<font face='Courier'>tile_registry</font>, and the two halves of the pipeline "
          "disagree about the tile name: Stage 1 writes "
          "<font face='Courier'>24_10.jpeg</font>, packaging writes "
          "<font face='Courier'>24_10</font>. Both sides go through "
          "<font face='Courier'>backend/slide_naming.py</font> so a key built from one "
          "can match a row registered from the other."),
        H2("A grep will not find every reader"),
        P("<font face='Courier'>app/hpc_chat_handlers_v23.py:334</font> does not query tables by "
          "name. It enumerates the whole database with "
          "<font face='Courier'>insp.get_table_names()</font>, keeps every table carrying an "
          "<font face='Courier'>hpc_id</font> or <font face='Courier'>dominant_hpc</font> column — "
          "skipping only <font face='Courier'>hpc_dictionary</font> and "
          "<font face='Courier'>h_latent_vectors</font> — and renders up to five matching rows "
          "straight to the user. So such a table is answered out of the chatbot without ever "
          "appearing in a query, and a grep for its name finds nothing. That is how "
          "<font face='Courier'>slide_hpc_membership</font> looked unreferenced while serving "
          "19,493 stale rows from an older cohort. <b>Before concluding a table is dead, check "
          "whether it has an <font face='Courier'>hpc_id</font> column.</b>"),
        H2("Aggregates are what the UI actually reads"),
        P("<font face='Courier'>hpl_profile_proportion</font> and "
          "<font face='Courier'>hpl_profile_summary</font> are derived from the same "
          "assignment CSV as <font face='Courier'>tile_registry</font>, and they — not "
          "<font face='Courier'>tile_registry</font> — are what the chatbot and the HPC "
          "panels query. Writing tiles without refreshing the aggregates leaves the UI "
          "internally inconsistent with nothing to signal it."),
        note("Two schema facts worth knowing before extending this. "
             "<font face='Courier'>hpl_profile_summary</font>'s uniqueness constraint is "
             "on <font face='Courier'>(samples, slides)</font> and does <b>not</b> include "
             "<font face='Courier'>dataset_id</font>, so two cohorts holding the same "
             "slide id cannot both have a summary row. And "
             "<font face='Courier'>slide_hpc_membership</font> has no "
             "<font face='Courier'>dataset_id</font> column at all, so it cannot be "
             "scoped per cohort even in principle.", WARN),
        H2("Schema provenance"),
        P("Only nine of the seventeen tables had a "
          "<font face='Courier'>CREATE TABLE</font> anywhere in git until 2026-08-26, and "
          "<font face='Courier'>dataset_id</font> — the column every cohort guard is "
          "scoped by — appeared in no migration at all. "
          "<font face='Courier'>backend/migrate_kb_base_tables.sql</font> now covers the "
          "missing eight. <font face='Courier'>schema.sql</font> in the repository root "
          "is a stale 2025-10-23 dump that disagrees with the live "
          "<font face='Courier'>tile_registry</font> on its primary key and on "
          "<font face='Courier'>hpc_id</font>'s type; it carries a header saying so."),
        code("psql -h &lt;socket&gt; -d hpl_kb -f backend/migrate_all.sql\n"
             "# idempotent; runs every migration in dependency order"),
    ]

    # ---- classifier ------------------------------------------------------
    # Flows rather than breaking: forcing this onto a new page left the
    # Knowledge Bank page four-fifths empty.
    s += [Spacer(1, 10), H1("The classifier"), ClassifierFlow(CONTENT_W)]

    s += [
        Spacer(1, 6),
        P("Search is <b>faiss-only and exact</b> — "
          "<font face='Courier'>IndexFlatL2</font>, no numpy fallback and no approximate "
          "index. An approximate index was measured against the real reference: it "
          "agreed on the nearest neighbour 33% of the time for no speed gain at this "
          "reference size, which would have put an approximation inside the one number "
          "the pipeline is judged on."),
    ]

    s.append(table([
        ["Measurement", "What it actually asks", "Result"],
        ["Leave-one-out",
         "Does k-NN recover a LATTICeA tile's own label, given the rest of LATTICeA "
         "(slide-mates included)?", "<b>97.23%</b> at the settled vote"],
        ["Closed-book slide holdout",
         "A LATTICeA slide the reference has never seen — 20 slides, 16,468 tiles held out",
         "<b>97.33%</b> ± 0.60 between-slide SD"],
        ["TCGA acceptance test",
         "Do we reproduce <font face='Courier'>sc.tl.ingest</font>'s cross-cohort "
         "transfer of LATTICeA clusters onto TCGA? 499,108 tiles",
         "<b>99.619%</b> agreement"],
        ["A new cohort (e.g. Radiogenomics)",
         "No ground truth exists. Probe by distance instead — "
         "<font face='Courier'>backend/cohort_shift.py</font>",
         "unmeasured by construction"],
    ], [CONTENT_W * 0.22, CONTENT_W * 0.52, CONTENT_W * 0.26]))

    s += [
        Spacer(1, 9),
        P("<b>The settled vote:</b> "
          "<font face='Courier'>--k 10 --distance-weighted --distance-power 3</font>, "
          "re-voting at <font face='Courier'>k=25</font> for tiles whose "
          "<font face='Courier'>vote_margin</font> falls below 0.15. It searches once at "
          "the wider k and votes on a prefix, so the base vote and the re-vote see the "
          "same neighbours by construction."),
        note("Leave-one-out accuracy and the acceptance test move in <i>opposite</i> "
             "directions on purpose. The acceptance test asks whether we reproduce "
             "<font face='Courier'>sc.tl.ingest</font>, which is an unweighted majority "
             "vote — so distance weighting makes the classifier deliberately less like "
             "<font face='Courier'>ingest</font>. Run the acceptance test twice: at the "
             "default, to confirm the ≥99% gate that protects against projection and "
             "centering bugs; and at the tuned preset, to see the size of the intended "
             "divergence. A drop at the <i>default</i> is a real defect.", ACCENT),
        H2("Confidence is calibrated, and Stage 6 can use it"),
        P("<font face='Courier'>vote_margin</font> stratifies accuracy cleanly even "
          "off-reference: on held-out slides, tiles above margin 0.75 were 100% correct "
          "(70% of all tiles) while those below 0.10 were 59.4%. Stage 6's "
          "<font face='Courier'>min_margin</font> excludes low-confidence tiles from the "
          "per-slide aggregates only — <font face='Courier'>tile_registry</font> keeps "
          "every tile's own <font face='Courier'>hpc_id</font> and margin regardless."),
    ]

    # ---- failure modes ---------------------------------------------------
    s += [PageBreak(), H1("The failure mode this codebase is written against")]
    s += [
        P("Almost nothing here fails by crashing. A wrong reference, a naming mismatch, "
          "a shard carrying its own mean, a half-merged output, a stale aggregate — each "
          "produces a file or a table of the right shape and dtype with no missing "
          "values. It passes every completeness check while every cluster id is attached "
          "to the wrong tile."),
        P("Everything below follows from that, and is worth preserving in anything added "
          "to this pipeline."),
    ]

    s.append(table([
        ["Guard", "Where", "What it refuses"],
        ["95% match rate",
         "<font face='Courier'>load_hpc_assignments.py</font>",
         "A load where the CSV and the registry disagree about slide naming for all but "
         "a handful of tiles. The number it guards against is not 0% — that is obvious — "
         "but the 3% that reads as “it worked” in a summary line."],
        ["Unknown cluster ids",
         "Stage 6, CLI and endpoint",
         "Cluster ids with no <font face='Courier'>hpc_dictionary</font> row. Those tiles "
         "would render with no pattern or malignancy annotation."],
        ["Cohort collision",
         "<font face='Courier'>register_dataset.py</font>",
         "A tile or slide whose key already belongs to a different "
         "<font face='Courier'>dataset_id</font>. Refused outright, never reassigned — "
         "two cohorts claiming one tile means one of them is misidentified."],
        ["Ambiguous slide file",
         "Stage 5",
         "A slide id matching more than one file. Registering either would mean being "
         "silently wrong about which physical slide a cohort's tiles came from."],
        ["Atomic publish",
         "<font face='Courier'>make_hpl_hdf5.py</font>",
         "A half-written <font face='Courier'>.h5</font> occupying the final path. Output "
         "is staged to <font face='Courier'>.partial</font> and renamed only when whole."],
        ["Flush before checkpoint",
         "packaging",
         "Recording tiles as done before their pixels are durable — which left "
         "permanently zero-filled holes that resume would skip past forever."],
        ["<font face='Courier'>--centering query</font> sharding",
         "<font face='Courier'>assign_hpc_clusters.py</font>",
         "Sharding without a shared precomputed mean. The mean is over <i>all</i> "
         "queries, so chunking silently changes every label."],
        ["Preview before commit",
         "Stages 5 and 6",
         "Any automatic write to the shared Knowledge Bank. A dry run has to be pulled "
         "up first, and the endpoint calls the same functions the CLI does rather than "
         "reimplementing the guards more loosely."],
    ], [CONTENT_W * 0.20, CONTENT_W * 0.22, CONTENT_W * 0.58]))

    s += [
        Spacer(1, 9),
        note("Two rules for anyone extending this. Prefer a loud refusal at submit time "
             "over a plausible result later. And when adding a test, make it prove the "
             "guard can <i>fail</i> — several bugs here were found by checking that a "
             "validator could come out bad, not that it came out good.", OK),
    ]

    # ---- operations ------------------------------------------------------
    s += [PageBreak(), H1("Operations")]
    s += [
        H2("Running it"),
        code("# on the HPC (inside the VPN)\n"
             "cd backend &amp;&amp; python tile_server_v2_.py        # FastAPI on :8000\n"
             "\n"
             "# on the laptop, with a tunnel to the login node\n"
             "ssh -N -L 8000:localhost:8000 -L 5432:localhost:5432 beatson-hpc\n"
             "streamlit run app/app_v28.py                   # talks to localhost:8000"),
        H2("Deploying a change"),
        P("<font face='Courier'>backend/</font> must be copied to the cluster and the "
          "server restarted; <font face='Courier'>app/</font> runs on the laptop. They "
          "are separate processes and neither picks up the other's changes. Two "
          "additional couplings are easy to miss: the package list for the Singularity "
          "container lives in <font face='Courier'>submit_feature_extraction.py</font>, "
          "so that file and <font face='Courier'>--bootstrap-extras</font> must always "
          "move together; and the encoder patch is mirrored in "
          "<font face='Courier'>backend/patches/hpl-encode-io.patch</font> because the "
          "HPC has its own clone of the HPL repository that the git subtree does not "
          "update."),
        H2("Schema migration"),
        code("psql -h /nfs/home/users/vpandya/databases/postgres_hpl_kb/socket -d hpl_kb \\\n"
             "     -f backend/migrate_all.sql\n"
             "# idempotent — a no-op against an up-to-date database"),
        H2("Tests"),
        code("python3 -m pytest backend/tests -q\n"
             "python3 backend/tests/test_register_dataset.py   # standalone, no pytest"),
        P("Every suite also runs standalone, which is how they run on the cluster where "
          "pytest may not be installed."),
        H2("Known open items"),
    ]

    s.append(table([
        ["Item", "State"],
        ["<font face='Courier'>tile_hpc_heatmap</font> has no writer",
         "The server reads all 149 MB of it at startup and merges its 71 "
         "<font face='Courier'>p_hpc_*</font> columns into the tile overlay. The k-NN "
         "classifier does not produce a 71-class distribution — it produces a top-1 label "
         "and a margin. Filling this is a modelling decision, not plumbing."],
        ["Four cluster reference tables have no trustworthy DDL in git",
         "<font face='Courier'>\\d hpc_dictionary</font>, "
         "<font face='Courier'>\\d hpc_malignant_details</font>, "
         "<font face='Courier'>\\d hpc_non_malignant_details</font>, "
         "<font face='Courier'>\\d hpc_survival_analysis</font> and "
         "<font face='Courier'>\\d+ slide_metadata</font> have never been captured. "
         "Until they are, a fresh checkout still cannot build a complete schema."],
        ["The viewer assumes one scan geometry for every cohort",
         "<font face='Courier'>TILE_SIZE_NATIVE</font> is computed from a hard-coded "
         "0.252 mpp. <font face='Courier'>dataset_config</font> and "
         "<font face='Courier'>wsi_metadata</font> now carry the real numbers per cohort; "
         "using them is a separate change with real risk to the viewer."],
        ["No cohort has been registered against the <i>live</i> database",
         "The path was verified end to end against a local PostgreSQL 16.2 — empty database, "
         "migrate, register, load — which found two bugs SQLite could not: a fresh schema that "
         "still would not build, and <font face='Courier'>dataset_id</font> never being set on "
         "the aggregates although it is NOT NULL there. Both fixed. The live database has real "
         "slide names and real scale, so preview the first cohort before committing it."],
    ], [CONTENT_W * 0.30, CONTENT_W * 0.70]))

    return s


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build(story())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
