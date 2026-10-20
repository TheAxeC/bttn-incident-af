"""End-to-end BTTN pipeline overview, drawn in the BTTR-paper visual language
(ECG traces, 3-D spectrogram cube, factor-matrix parallelograms, conv/attention
box, transparent clinical-fusion node, Cox survival curve). One left-to-right
flow with the six stages a reader follows: raw ECG, spectral tensor, interpretable
block factors, temporal pattern extraction, transparent clinical fusion, survival
prediction. -> results/pipeline_overview.png. Pure matplotlib (no torch, no data).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle, Circle

RED, BLUE, GREEN, CORE = "#c0392b", "#2e6da4", "#27ae60", "white"
PURPLE, REDH, GOLD = "#7d5ba6", "#e74c3c", "#caa84a"


def cube(ax, ox, oy, w, h, d, label="", dim=None, top="#cfe3cf", front="#bcd8bc", side="#9cc39c"):
    sx, sy = 0.42 * d, 0.50 * d
    ax.add_patch(Polygon([(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h)], fc=front, ec="k", lw=1.1))
    ax.add_patch(Polygon([(ox, oy + h), (ox + w, oy + h), (ox + w + sx, oy + h + sy), (ox + sx, oy + h + sy)], fc=top, ec="k", lw=1.1))
    ax.add_patch(Polygon([(ox + w, oy), (ox + w + sx, oy + sy), (ox + w + sx, oy + h + sy), (ox + w, oy + h)], fc=side, ec="k", lw=1.1))
    if label:
        ax.text(ox + w / 2 + sx / 2, oy + h / 2 + sy / 2, label, ha="center", va="center", fontsize=13, style="italic")
    if dim:
        ax.text(ox - 0.12, oy + h / 2, dim[0], ha="right", va="center", fontsize=8.5)
        ax.text(ox + w / 2, oy - 0.16, dim[1], ha="center", va="top", fontsize=8.5)
        ax.text(ox + w + sx / 2 + 0.05, oy + h + sy + 0.10, dim[2], ha="center", va="bottom", fontsize=8.5)


def rrect(ax, cx, cy, w, h, ang, color, label="", dim="", hatch="////", fs=9):
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T) + [cx, cy]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="k", lw=1.1, alpha=0.92, hatch=hatch))
    if label:
        ax.text(cx, cy + h / 2 * abs(np.cos(a)) + 0.42, label, ha="center", va="bottom", fontsize=fs)
    if dim:
        ax.text(cx, cy - h / 2 * abs(np.cos(a)) - 0.40, dim, ha="center", va="top", fontsize=7.5, color="#333")


def arrow(ax, p, q, color="#333", lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=17, lw=lw, color=color, shrinkA=0, shrinkB=0))


def stage_title(ax, xc, n, text, color):
    """Numbered circle above the panel, short wrapped label centered below it."""
    ax.text(xc, 6.85, "%d" % n, ha="center", va="center", fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.30", fc=color, ec="k", lw=1.0))
    ax.text(xc, 6.25, text, ha="center", va="top", fontsize=9.2, fontweight="bold", color=color)


fig, ax = plt.subplots(figsize=(16, 6.2))
ax.set_xlim(0, 33)
ax.set_ylim(-0.6, 7.2)
ax.axis("off")

# ---------------- Stage 1: raw 12-lead ECG ----------------
stage_title(ax, 1.9, 1, "12-lead ECG", RED)
ecg_x = np.linspace(0, 1, 200)
for i in range(4):
    base = 1.4 + 1.05 * i
    beat = np.zeros_like(ecg_x)
    for c in (0.25, 0.55, 0.85):
        beat += 0.9 * np.exp(-((ecg_x - c) * 60) ** 2)                     # R spike
        beat += 0.14 * np.exp(-((ecg_x - (c - 0.09)) * 25) ** 2)          # P wave
        beat += 0.20 * np.exp(-((ecg_x - (c + 0.13)) * 18) ** 2)          # T wave
    ax.plot(0.3 + ecg_x * 3.2, base + 0.42 * beat, color="#333", lw=1.0)
ax.add_patch(Rectangle((0.25, 1.15), 3.35, 4.6, fill=False, ec="#bbb", lw=0.8))
ax.text(1.9, 0.75, "10 s per lead", ha="center", va="top", fontsize=7.5, color="#666")
arrow(ax, (3.7, 3.4), (5.0, 3.4))

# ---------------- Stage 2: spectral tensor ----------------
stage_title(ax, 6.7, 2, "spectral tensor", GREEN)
cube(ax, 5.6, 2.3, 2.1, 2.5, 1.5, label=r"$\mathcal{X}$", dim=("freq", "time", "lead"))
ax.text(6.7, 1.75, r"$F{\times}T{\times}L$", ha="center", va="top", fontsize=8.5, color="#555")
ax.text(6.7, 1.25, "STFT, per lead", ha="center", va="top", fontsize=7.5, color="#666")
arrow(ax, (9.4, 3.4), (10.7, 3.4))

# ---------------- Stage 3: interpretable block factors ----------------
stage_title(ax, 11.9, 3, "interpretable\nblock factors", BLUE)
rrect(ax, 11.9, 4.2, 0.55, 1.9, -22, RED, label=r"$\mathbf{A}_k$", dim=r"freq $F{\times}r_f$")
rrect(ax, 11.5, 2.4, 1.5, 1.0, -28, BLUE, label="", dim=r"$\mathbf{C}_k:$ lead $L{\times}r_l$")
ax.add_patch(Rectangle((12.35, 3.15), 0.5, 0.4, fc=CORE, ec="k", lw=1.1))
ax.text(12.6, 3.35, r"$\mathbf{G}_k$", ha="center", va="center", fontsize=9)
ax.text(11.9, 1.35, "K blocks; read\ndirectly (glass box)", ha="center", va="top", fontsize=7.5, color=BLUE)
arrow(ax, (13.6, 3.4), (14.9, 3.4))

# ---------------- Stage 4: temporal pattern extraction ----------------
stage_title(ax, 16.5, 4, "temporal\npattern extraction", GREEN)
for i in range(3):
    xs = np.linspace(0, 1, 60)
    ax.plot(15.6 + xs * 1.8, 3.7 + 0.30 * i + 0.16 * np.sin((6 + 4 * i) * xs) * np.exp(-((xs - .5) * 3) ** 2),
            color=GREEN, lw=1.5)
ax.add_patch(Rectangle((15.55, 3.55), 1.9, 1.05, fill=False, ec="#888", lw=0.8))
ax.text(16.5, 4.85, r"$\Psi_k$ multi-scale conv", ha="center", fontsize=8, color=GREEN)
ax.add_patch(Rectangle((15.55, 2.35), 1.9, 0.95, fc="#f3e9c7", ec=GOLD, lw=1.1))
ax.text(16.5, 2.82, "attention\npooling", ha="center", va="center", fontsize=8)
ax.text(16.5, 1.55, "when each block\nresponds", ha="center", va="top", fontsize=7.5, color="#666")
arrow(ax, (17.7, 3.4), (19.0, 3.4))

# ---------------- Stage 5: transparent clinical fusion ----------------
stage_title(ax, 21.0, 5, "transparent\nclinical fusion", PURPLE)
ax.add_patch(Rectangle((19.5, 3.75), 2.3, 1.4, fc=PURPLE, ec="k", lw=1.1))
ax.text(20.65, 4.45, "BTTN head\n" r"$\eta_{\mathrm{ECG}}$", ha="center", va="center", fontsize=8.5, color="white")
ax.add_patch(Rectangle((19.5, 1.95), 2.3, 1.2, fc="#e8e2f0", ec=PURPLE, lw=1.1))
ax.text(20.65, 2.55, "age, sex,\nECG intervals\n" r"$\mathbf{w}^{\top}\mathbf{c}$", ha="center", va="center", fontsize=7.8)
ax.add_patch(Circle((22.8, 3.55), 0.34, fc="white", ec="k", lw=1.3))
ax.text(22.8, 3.55, "+", ha="center", va="center", fontsize=15)
arrow(ax, (21.8, 4.45), (22.55, 3.75), lw=1.6)
arrow(ax, (21.8, 2.55), (22.55, 3.35), lw=1.6)
ax.text(22.8, 1.55, "additive term\nbypasses blocks", ha="center", va="top", fontsize=7.5, color=PURPLE)
arrow(ax, (23.2, 3.55), (24.5, 3.55))

# ---------------- Stage 6: survival prediction ----------------
stage_title(ax, 26.95, 6, "survival\nprediction", REDH)
xs = np.linspace(0, 1, 60)
ax.plot(25.2 + xs * 3.4, 2.9 + 1.5 * (1 - np.exp(-3 * xs)), color=REDH, lw=2.4)
ax.add_patch(Rectangle((25.1, 2.7), 3.7, 1.95, fill=False, ec="#aaa", lw=0.8))
ax.text(26.95, 2.35, "Cox partial likelihood", ha="center", va="top", fontsize=8, color=REDH)
ax.text(26.95, 1.85, "one model, risk at\n1 / 2 / 5 yr", ha="center", va="top", fontsize=7.8, color="#555")

# ---------------- retained-glass-box band ----------------
fig.text(0.5, 0.03,
         "Interpretability is preserved end to end: the per-block frequency factor "
         r"$\mathbf{A}_k$ and lead factor $\mathbf{C}_k$ (stage 3) are read directly, and the "
         "clinical term (stage 5) is added AFTER the block layer, so fusion leaves the interpretable "
         "factors and the faithfulness probe unchanged.",
         ha="center", fontsize=9.0,
         bbox=dict(boxstyle="round,pad=0.5", fc="#fbf3e7", ec=GOLD))

fig.suptitle("BTTN pipeline: from the sinus-rhythm ECG to an interpretable survival risk", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])

from core import config
fig.savefig(config.RESULTS_DIR / "pipeline_overview.png", dpi=160, bbox_inches="tight")
print("[saved] results/pipeline_overview.png")
