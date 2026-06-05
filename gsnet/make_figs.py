#
# Generate conference-style (Times, clean) figures for the GS-Net meeting/paper.
# Run:  python make_figs.py      ->  fig_tsweep.pdf, fig_densify.pdf, fig_pseudogt.pdf
# Then \includegraphics{...} them in meeting_results.tex.
#
# All data are the validated (good) results; no external files needed.
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- conference style -------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
GS = "#1f77b4"      # GS-Net blue
ACC = "#d62728"     # accent / best-marker


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---- Fig 1: CARLA T-sweep --------------------------------------------
def fig_tsweep():
    T = [3, 5, 8, 12, 16]
    psnr = [26.14, 27.19, 26.11, 26.57, 26.40]
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.plot(T, psnr, "-o", color=GS)
    bi = psnr.index(max(psnr))
    ax.plot(T[bi], psnr[bi], "o", color=ACC, ms=7, zorder=5)
    ax.annotate("best ($T{=}5$)", (T[bi], psnr[bi]),
                textcoords="offset points", xytext=(6, -2), color=ACC, fontsize=8)
    ax.set_xlabel("Number of expansion heads $T$")
    ax.set_ylabel("SSE PSNR (dB)")
    ax.set_xticks(T)
    _clean(ax)
    fig.savefig("fig_tsweep.pdf")
    plt.close(fig)


# ---- Fig 2: CSE densification sweet spot -----------------------------
def fig_densify():
    x = [0, 2000, 5000, 10000, 15000]
    d = [1.28, 1.98, -0.01, 0.04, 0.02]
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.plot(x, d, "-o", color=GS)
    bi = d.index(max(d))
    ax.plot(x[bi], d[bi], "o", color=ACC, ms=7, zorder=5)
    ax.annotate("sweet spot", (x[bi], d[bi]),
                textcoords="offset points", xytext=(6, -1), color=ACC, fontsize=8)
    ax.set_xlabel("densify_until_iter")
    ax.set_ylabel(r"$\Delta$PSNR (GS-Net $-$ base)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=30, ha="right")
    _clean(ax)
    fig.savefig("fig_densify.pdf")
    plt.close(fig)


# ---- Fig 3: pseudo-GT graceful degradation (Reviewer C) --------------
def fig_pseudogt():
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 2.3))
    # (a) supervision quality
    it = [5, 10, 20, 30]
    pa = [23.80, 24.13, 23.96, 24.46]
    a.plot(it, pa, "-o", color=GS)
    a.set_xlabel("Pseudo-GT supervision iter (k)")
    a.set_ylabel("PSNR (dB)")
    a.set_xticks(it)
    a.set_title("(a) vs. supervision", fontsize=9)
    _clean(a)
    # (b) density
    dn = [10, 25, 50, 100]
    pb = [23.64, 24.00, 24.03, 24.46]
    b.plot(dn, pb, "-o", color=GS)
    b.set_xlabel("Pseudo-GT density kept (%)")
    b.set_xticks(dn)
    b.set_title("(b) vs. density", fontsize=9)
    _clean(b)
    for ax in (a, b):
        ax.set_ylim(23.4, 24.7)
    fig.tight_layout(w_pad=1.5)
    fig.savefig("fig_pseudogt.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_tsweep()
    fig_densify()
    fig_pseudogt()
    print("wrote fig_tsweep.pdf, fig_densify.pdf, fig_pseudogt.pdf")
