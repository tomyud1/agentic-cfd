"""Report figures (matplotlib -> inline SVG). Static, light-surface, print-friendly.

Sizes: the report column is about 9.2 in wide and figures are shown at their natural size (never
enlarged), so the font sizes set here are the sizes the reader sees.
  figure()          one panel, 6.4 x 3.8 in
  wide(ncols)       2-3 panels side by side, 9.2 in wide (the full column)
  sketch()          problem sketch, 7.5 x 5.5 in, no axes
Anything wider than the column is scaled down and its text shrinks: to_svg() refuses it.

to_svg() also checks the layout and raises FigureLayoutError if text overlaps other text, a data
line, an arrow or a solid shape, or if a legend covers data. Fix the layout (move labels, use
annotate() with an offset, legend outside, fewer ticks). Only for an overlap that is intentional,
pass allow=("substring of the issue text", ...).
One legend for several panels: fig.legend(handles, labels, loc="outside lower center", ncol=3) puts
it under the panels (the figure must use layout="constrained", as figure() and wide() do) and passes
the check. The check sees lines, patches and arrows; lines drawn as a LineCollection (a grid of many
segments) are not seen, so look at such a figure yourself (CFD_FIG_PNG_DIR).
Set CFD_FIG_PNG_DIR=<dir> to also write every figure as a PNG (to look at it).
"""
import io
import logging
import os
import re
from pathlib import Path

import matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, fixed order
COLUMN_IN = 9.2                                         # report column width, inches

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    # Text as vector paths: renders identically in every browser and in print.
    # Math ($...$) in Computer Modern, matching the report's MathJax equations.
    "lines.linewidth": 2, "svg.fonttype": "path", "svg.hashsalt": "agentic-cfd", "mathtext.fontset": "cm",
})


class FigureLayoutError(ValueError):
    pass


def figure(w=6.4, h=3.8):
    """One panel."""
    return plt.subplots(figsize=(w, h), layout="constrained")


def wide(ncols=2, h=3.6, **kw):
    """Panels side by side, filling the report column."""
    return plt.subplots(1, ncols, figsize=(COLUMN_IN, h), layout="constrained", **kw)


def sketch(w=7.5, h=5.5):
    """Problem sketch: equal aspect, no axes."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---- layout check -----------------------------------------------------------------------------
_drawn = None                      # texts drawn during a check: list of (Text, Bbox)
_text_draw = Text.draw


def _recording_draw(self, renderer):
    _text_draw(self, renderer)
    if _drawn is not None and self.get_visible() and self.get_text().strip():
        _drawn.append((self, Text.get_window_extent(self, renderer)))   # text only, not an annotation arrow


Text.draw = _recording_draw


def _shrink(b, px=1.0):
    return Bbox.from_extents(b.x0 + px, b.y0 + px, b.x1 - px, b.y1 - px)


def _label(t):
    s = re.sub(r"\s+", " ", t.get_text()).strip()
    return f'"{s[:40]}"'


def _parts(path):
    """A path as separate polylines (CLOSEPOLY/MOVETO codes would otherwise add bogus segments)."""
    return [MPath(v) for v in path.to_polygons(closed_only=False) if len(v) >= 1]


def _hits(path, box, filled):
    for part in _parts(path):
        if len(part.vertices) == 1:
            if box.contains(*part.vertices[0]):
                return True
        elif part.intersects_bbox(box, filled=filled):
            return True
    return False


def check_layout(fig):
    """List of layout problems (empty = fine). See the module docstring."""
    global _drawn
    _drawn = []
    try:
        fig.canvas.draw()
        texts = [(t, _shrink(b)) for t, b in _drawn if b.width > 2 and b.height > 2]
    finally:
        _drawn = None
    renderer = fig.canvas.get_renderer()
    issues = []

    # 1. text on text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (t1, b1), (t2, b2) = texts[i], texts[j]
            if b1.overlaps(b2):
                issues.append(f"text {_label(t1)} overlaps text {_label(t2)}")

    for ax in fig.axes:
        abox = ax.get_window_extent(renderer)
        area = max(abox.width * abox.height, 1.0)
        shapes = []                # (artist, display path, filled)
        for ln in ax.lines:
            if ln.get_visible() and len(ln.get_xdata()):
                shapes.append((ln, ln.get_transform().transform_path(ln.get_path()),
                               False, ln.get_linestyle() in ("None", "none", " ", "")))
        for p in ax.patches:
            if not p.get_visible() or p is ax.patch:
                continue
            path = p.get_transform().transform_path(p.get_path())
            if isinstance(p, FancyArrowPatch):
                shapes.append((p, path, False, False))
            else:
                # Hatched or small solid shapes (walls, bodies): text must stay off them entirely.
                # Any other shape (outlines, large backgrounds): text must not cross its edge.
                ext = path.get_extents()
                solid = p.get_fill() and p.get_alpha() != 0
                body = bool(p.get_hatch()) or (solid and ext.width * ext.height < 0.15 * area)
                shapes.append((p, path, body, False))
        for t in ax.texts:
            if isinstance(t, Annotation) and t.arrow_patch is not None and t.get_visible():
                ap = t.arrow_patch
                shapes.append((ap, ap.get_transform().transform_path(ap.get_path()), False, False))

        # 2. text on a line, arrow or solid shape drawn in this axes
        for t, b in texts:
            clip = Bbox.intersection(_shrink(b, -3), abox)      # keep ~2 px of clear space around text
            if clip is None or clip.width < 1 or clip.height < 1:
                continue
            for art, path, filled, markers_only in shapes:
                if isinstance(t, Annotation) and art is t.arrow_patch:
                    continue
                if markers_only:
                    hit = any(clip.contains(x, y) for x, y in path.vertices)
                else:
                    hit = _hits(path, clip, filled)
                if hit:
                    kind = "arrow" if isinstance(art, FancyArrowPatch) else \
                           "shape" if isinstance(art, Patch) else "line"
                    lab = art.get_label() if not str(art.get_label()).startswith("_") else ""
                    issues.append(f"text {_label(t)} overlaps a {kind}{' ' + repr(lab) if lab else ''}")
                    break

        # 3. legend covering data
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            lb = _shrink(leg.get_window_extent(renderer), 2)
            for ln in ax.lines:
                if ln.get_visible() and len(ln.get_xdata()) and not str(ln.get_label()).startswith("_"):
                    path = ln.get_transform().transform_path(ln.get_path())
                    if _hits(path, lb, False) or any(lb.contains(x, y) for x, y in path.vertices):
                        issues.append(f"legend covers data line {ln.get_label()!r} (move it outside or to an empty corner)")
                        break
    return issues


# ---- export -----------------------------------------------------------------------------------
_png_count = 0


def to_svg(fig, name=None, allow=()):
    """Check the layout, then return the figure as an inline SVG string (and close it)."""
    global _png_count
    issues = [i for i in check_layout(fig) if not any(a in i for a in allow)]
    w_in = fig.get_size_inches()[0]
    if w_in > COLUMN_IN + 0.05:
        issues.append(f"figure is {w_in:.1f} in wide; the report column is {COLUMN_IN} in, so it would be "
                      f"shrunk and its text would become smaller (use plots.wide() or fewer panels)")
    png_dir = os.environ.get("CFD_FIG_PNG_DIR")
    if png_dir:
        _png_count += 1
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(png_dir) / f"{name or f'fig{_png_count:02d}'}.png", dpi=110, bbox_inches="tight")
    if issues:
        plt.close(fig)
        raise FigureLayoutError(f"figure {name or ''} layout problems:\n  - " + "\n  - ".join(issues))
    buf = io.StringIO()
    # no date in the file and ids that do not change from run to run (svg.hashsalt, set above): rendering the
    # same data twice gives the same bytes, so "did a report change?" is a plain file comparison
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]
