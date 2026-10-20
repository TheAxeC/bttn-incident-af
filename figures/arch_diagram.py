"""Tensor-style schematic of BTTN, drawn in the BTTR-paper visual language
(3-D tensor cubes, factor-matrix parallelograms, cores, dimension labels).
Row 1 recaps BTTR; row 2 shows BTTN and what changes. -> results/bttr_vs_bttn.png
Pure matplotlib (no torch).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle

RED, BLUE, GREEN, CORE = "#c0392b", "#2e6da4", "#27ae60", "white"
PURPLE, REDH = "#7d5ba6", "#e74c3c"


def cube(ax, ox, oy, w, h, d, label="", dim=None, top="#dadada", front="#bcbcbc", side="#9a9a9a"):
    sx, sy = 0.42 * d, 0.50 * d
    ax.add_patch(Polygon([(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h)], fc=front, ec="k", lw=1.1))
    ax.add_patch(Polygon([(ox, oy + h), (ox + w, oy + h), (ox + w + sx, oy + h + sy), (ox + sx, oy + h + sy)], fc=top, ec="k", lw=1.1))
    ax.add_patch(Polygon([(ox + w, oy), (ox + w + sx, oy + sy), (ox + w + sx, oy + h + sy), (ox + w, oy + h)], fc=side, ec="k", lw=1.1))
    if label:
        ax.text(ox + w / 2 + sx / 2, oy + h / 2 + sy / 2, label, ha="center", va="center", fontsize=11, style="italic")
    if dim:
        ax.text(ox - 0.15, oy + h / 2, dim[0], ha="right", va="center", fontsize=8.5)          # I1 (left)
        ax.text(ox + w / 2, oy - 0.18, dim[1], ha="center", va="top", fontsize=8.5)             # I2 (bottom)
        ax.text(ox + w + sx / 2 + 0.05, oy + h + sy + 0.12, dim[2], ha="center", va="bottom", fontsize=8.5)  # I3 (top)


def rrect(ax, cx, cy, w, h, ang, color, label="", dim="", hatch="////", fs=9):
    a = np.deg2rad(ang); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T) + [cx, cy]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="k", lw=1.1, alpha=0.92, hatch=hatch))
    if label:
        ax.text(cx, cy + h / 2 * abs(np.cos(a)) + 0.55, label, ha="center", va="bottom", fontsize=fs)
    if dim:
        ax.text(cx, cy - h / 2 * abs(np.cos(a)) - 0.5, dim, ha="center", va="top", fontsize=7.5, color="#333")


def arrow(ax, p, q, color="#333", lw=1.7, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=15, lw=lw, color=color, shrinkA=0, shrinkB=0))


def block(ax, x, y, kind):
    """one block: red top factor, blue bottom factor, core, and a 'time' element
    (green score vector for BTTR; green conv-kernel bank for BTTN)."""
    rrect(ax, x + 0.15, y + 1.55, 0.55, 2.0, -22, RED,
          label=(r"$\mathbf{P}^{(2)}_k$" if kind == "bttr" else r"$\mathbf{A}_k$"),
          dim=(r"$I_3\times R^k_3$" if kind == "bttr" else r"freq $F\times r_f$"))
    rrect(ax, x - 0.35, y - 0.35, 1.5, 1.1, -28, BLUE,
          label="", dim=(r"$\mathbf{P}^{(1)}_k:\,I_2\times R^k_2$" if kind == "bttr" else r"$\mathbf{C}_k:\,$lead $L\times r_l$"))
    ax.add_patch(Rectangle((x + 0.55, y + 0.45), 0.5, 0.4, fc=CORE, ec="k", lw=1.1))
    ax.text(x + 0.8, y + 0.65, r"$\mathbf{G}_k$", ha="center", va="center", fontsize=9)
    if kind == "bttr":
        ax.add_patch(Rectangle((x + 1.25, y + 0.5), 1.4, 0.16, fc=GREEN, ec="k", lw=1.0))
        ax.text(x + 1.95, y + 0.9, r"$\mathbf{t}_k\in R^{I_1\times1}$", ha="center", fontsize=8.5)
        ax.text(x + 1.95, y + 0.25, "score vector\n(static)", ha="center", va="top", fontsize=7, color="#666")
    else:
        for i in range(3):                                   # multi-scale conv kernels
            xs = np.linspace(0, 1, 40)
            ax.plot(x + 1.3 + xs * 1.3, y + 0.45 + 0.18 * i + 0.12 * np.sin((6 + 4 * i) * xs) * np.exp(-((xs - .5) * 3) ** 2),
                    color=GREEN, lw=1.4)
        ax.add_patch(Rectangle((x + 1.28, y + 0.38), 1.34, 0.62, fill=False, ec="#888", lw=0.7))
        ax.text(x + 1.95, y + 1.12, r"$\Psi_k$  multi-scale", ha="center", fontsize=8.5, color=GREEN)
        ax.text(x + 1.95, y + 0.28, "conv along time\n(LEARNED kernels)", ha="center", va="top", fontsize=7, color="#666")


fig, axes = plt.subplots(2, 1, figsize=(15, 10));
for ax in axes:
    ax.set_xlim(0, 30); ax.set_ylim(-1.5, 7); ax.axis("off")

# ---------------- ROW 1: BTTR ----------------
ax = axes[0]
ax.text(0.2, 6.5, "BTTR (Faes 2022): supervised block-term tensor regression: linear, deflation-fit",
        fontsize=12.5, fontweight="bold", color=BLUE)
cube(ax, 1, 1.5, 2.2, 2.6, 1.6, label=r"$\mathcal{X}$", dim=(r"$I_1$", r"$I_2$", r"$I_3$"))
ax.text(4.6, 3.0, "=", fontsize=20, ha="center")
block(ax, 6.0, 2.2, "bttr")
ax.text(10.7, 3.0, r"$+\ \cdots\ +$", fontsize=15, ha="center")
block(ax, 12.5, 2.2, "bttr")
arrow(ax, (17.5, 3.0), (19.2, 3.0))
ax.text(22.2, 3.3, r"$\hat{y}=\sum_k b_k\,\mathbf{t}_k$", fontsize=13, ha="center", color=BLUE)
ax.text(22.2, 2.4, "LINEAR score\n(one LS target)", fontsize=8.5, ha="center", va="top", color="#555")
ax.add_patch(Rectangle((24.6, 1.8), 5.0, 2.4, fc="#dbe7f0", ec=BLUE, lw=1.4))
ax.text(27.1, 3.0, "interpretable\nblock maps  (yes)\n\nbut: linear,\nno time-to-event", ha="center", va="center", fontsize=9)

# ---------------- ROW 2: BTTN ----------------
ax = axes[1]
ax.text(0.2, 6.5, "BTTN (this work): same block STRUCTURE, now learnable, nonlinear, survival network",
        fontsize=12.5, fontweight="bold", color=GREEN)
cube(ax, 1, 1.5, 2.2, 2.6, 1.6, label=r"$\mathcal{X}$", dim=(r"freq", r"time", r"lead"),
     top="#cfe3cf", front="#bcd8bc", side="#9cc39c")
ax.text(0.2, 0.7, "spectrogram tensor", fontsize=8, color="#555")
ax.text(4.6, 3.0, r"$\mapsto$", fontsize=20, ha="center")
block(ax, 6.0, 2.2, "bttn")
ax.text(10.7, 3.0, r"$+\ \cdots\ +$", fontsize=15, ha="center")
block(ax, 12.5, 2.2, "bttn")
arrow(ax, (17.5, 3.0), (19.0, 3.0))
ax.add_patch(Rectangle((19.0, 2.2), 2.4, 1.7, fc="#f3e9c7", ec="#caa84a", lw=1.2))
ax.text(20.2, 3.05, "attention\nover blocks", ha="center", va="center", fontsize=8.5)
arrow(ax, (21.4, 3.0), (22.6, 3.0))
ax.add_patch(Rectangle((22.6, 2.2), 2.2, 1.7, fc=PURPLE, ec="k", lw=1.2))
ax.text(23.7, 3.05, "nonlinear\nhead", ha="center", va="center", fontsize=8.5, color="white")
arrow(ax, (24.8, 3.0), (26.0, 3.0))
# Cox hazard curve
xs = np.linspace(0, 1, 50)
ax.plot(26.1 + xs * 3.2, 2.4 + 1.2 * (1 - np.exp(-3 * xs)), color=REDH, lw=2)
ax.add_patch(Rectangle((26.1, 2.3), 3.2, 1.5, fill=False, ec="#aaa", lw=0.7))
ax.text(27.7, 1.95, "Cox survival risk\n1 / 2 / 5 yr", ha="center", va="top", fontsize=8, color=REDH)

# what-changes band
fig.text(0.5, 0.02,
         "WHAT CHANGES   BTTR to BTTN:   deflation/ACE fit becomes end-to-end gradient    |    "
         "static score vector $\\mathbf{t}_k$ becomes learned multi-scale temporal conv $\\Psi_k$    |    "
         "linear $\\hat y$ becomes attention + nonlinear head    |    least-squares becomes Cox survival (multi-horizon)    "
         "[ RETAINED: per-block freq $\\mathbf{A}_k$ and lead $\\mathbf{C}_k$ glass-box maps ]",
         ha="center", fontsize=9.2,
         bbox=dict(boxstyle="round,pad=0.5", fc="#fbf3e7", ec="#caa84a"))
fig.suptitle("From BTTR to BTTN", fontsize=15, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.06, 1, 0.96])
from core import config
fig.savefig(config.RESULTS_DIR / "bttr_vs_bttn.png", dpi=160, bbox_inches="tight")
print("[saved] results/bttr_vs_bttn.png")
