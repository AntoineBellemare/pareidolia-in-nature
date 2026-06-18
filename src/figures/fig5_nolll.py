"""Fig 5 (method convergence): the breadth gradient under both strategies.
muDelta (within-taxon stratified, iNat) per breadth class for the LLM-strip
corpus and for the raw myths (discrete matched-null frame). Both decay from
biome-specific to universal motifs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
LAD = EMB / "ladder"
OUT = ROOT / "paper/figures/fig5_nolll.png"

ORDER = ["SpecA", "Semi", "Universal"]
LABEL = {"SpecA": "Spec A\n(biome-specific)", "Semi": "Semi", "Universal": "Universal"}


def main():
    llm = pd.read_csv(EMB / "v3_breadth_sentpool_iNat.csv")
    llm_mu = (llm.groupby("breadth")["delta_strat"].mean() * 1000).reindex(ORDER)
    nm = pd.read_csv(LAD / "nolll_breadth.csv")
    nm = nm[nm.method == "raw"].set_index("bin").reindex(ORDER)

    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.plot(x, llm_mu.values, "-o", color="#3a7d44", lw=2.6, ms=9,
            label="remove the names (LLM-strip)", zorder=3)
    lo = nm["mu"] - nm["ci_lo"]; hi = nm["ci_hi"] - nm["mu"]
    ax.errorbar(x, nm["mu"].values, yerr=[lo.values, hi.values], color="#222",
                marker="s", ms=8, lw=2.6, capsize=4,
                label="hold the names constant (raw myths)", zorder=3)
    # green (LLM-strip) labels sit below their marker, black (matched) above,
    # with a fixed pixel offset so they never collide where the lines cross
    for xi in x:
        ax.annotate(f"{llm_mu.values[xi]:.2f}", (xi, llm_mu.values[xi]),
                    textcoords="offset points", xytext=(0, -12), ha="center",
                    va="top", fontsize=8.5, color="#3a7d44", fontweight="bold")
        ax.annotate(f"{nm['mu'].values[xi]:.2f}", (xi, nm["mu"].values[xi]),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    va="bottom", fontsize=8.5, color="#222", fontweight="bold")
    ax.axhline(0, color="#888", lw=0.7, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[b] for b in ORDER], fontsize=10)
    ax.set_xlabel("motif breadth  (specific $\\rightarrow$ universal)", fontsize=10)
    ax.set_ylabel(r"stratified $\mu\Delta$ ($\times 10^{-3}$)", fontsize=10)
    ax.set_title("Both methods recover the breadth gradient\n"
                 "alignment is strongest in biome-specific motifs and fades in universals",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(fontsize=9.5, loc="upper right", frameon=True)
    for s in ax.spines.values():
        s.set_color("#bbb")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
