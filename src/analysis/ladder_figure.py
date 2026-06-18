"""Figure for the ladder rung-1 decomposition.

Panel A: per-biome residualised Δ for the full original myth vs the three
         separated baselines (species / place / ethnonym).
Panel B: species-matched permutation null — observed Δ_full vs the
         within-species-block null 95% range, per biome.
Panel C: species-subspace projection triangulation — μΔ_full after
         removing 0/5/10/20 species directions (passed via PROJ).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for shared utils
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
LAD = EMB / "ladder"
OUT = ROOT / "paper/figures/figS_ladder_decomposition.png"

from make_phase2_figures import short_biome, biome_color

# projection μΔ (×10⁻³) for [base, 5, 10, 20] components removed — filled from stats stdout
PROJ = {"labels": ["0 (full)", "−5 dir", "−10 dir", "−20 dir"],
        "mu": None}  # set below if available


def main(proj_mu=None):
    dec = pd.read_csv(LAD / "stats_decomposition.csv")
    mn = pd.read_csv(LAD / "stats_species_matched_null.csv")

    order = dec.sort_values("delta_full", ascending=False)["biome"].tolist()
    dec = dec.set_index("biome").reindex(order)
    mn = mn.set_index("biome").reindex(order)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0],
                          width_ratios=[1.6, 1.0], hspace=0.45, wspace=0.28)
    fig.patch.set_facecolor("white")
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ---- Panel A: decomposition grouped bars ----
    axA.set_facecolor("white")
    y = np.arange(len(order)); bw = 0.2
    cols = {"full": "#222222", "species": "#cf6f3f",
            "place": "#5b8db8", "ethnonym": "#7cbe5e"}
    for k, name in enumerate(["full", "species", "place", "ethnonym"]):
        vals = dec[f"delta_{name}"].values * 1000
        axA.barh(y + (1.5 - k) * bw, vals, bw, label=name,
                  color=cols[name], edgecolor="#222", lw=0.3)
    axA.axvline(0, color="#666", lw=0.6)
    axA.set_yticks(y); axA.set_yticklabels([short_biome(b) for b in order],
                                            fontsize=8.5)
    axA.invert_yaxis()
    axA.set_xlabel(r"residualised marginal $\Delta$ ($\times 10^{-3}$), raw-Russian frame",
                   fontsize=9.5)
    axA.set_title("A. Per-biome decomposition: full original myth vs three separated baselines",
                   fontsize=11, fontweight="bold", loc="left")
    axA.legend(fontsize=9, loc="lower right", ncol=4, frameon=True)
    axA.tick_params(labelsize=8)
    for s in axA.spines.values(): s.set_color("#bbb")
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # ---- Panel B: matched-null ladder summary ----
    axB.set_facecolor("white")
    summ = pd.read_csv(LAD / "stats_matched_null_summary.csv")
    # order: species, ethnonym, place, joint
    o = {r["null"]: r for _, r in summ.iterrows()}
    seq = ["species", "ethnonym", "place", "joint"]
    seq = [s for s in seq if s in o]
    x = np.arange(len(seq)); bw = 0.38
    obs = [o[s]["obs"] for s in seq]
    nul = [o[s]["null_mean"] for s in seq]
    axB.bar(x - bw/2, obs, bw, color="#222222", edgecolor="#111", lw=0.4,
            label="observed Δ_full")
    axB.bar(x + bw/2, nul, bw, color="#c9a04a", edgecolor="#111", lw=0.4,
            label="matched-null mean")
    for i, s in enumerate(seq):
        axB.text(i + bw/2, nul[i] + 0.01,
                  f"{int(o[s]['n_sig'])}/{int(o[s]['n'])}\nsurvive",
                  ha="center", va="bottom", fontsize=7.5, color="#333")
    axB.set_xticks(x)
    axB.set_xticklabels([s.capitalize() for s in seq], fontsize=9)
    axB.set_ylabel(r"$\mu\Delta$ ($\times 10^{-3}$)", fontsize=9)
    axB.set_ylim(0, max(obs) * 1.35)
    axB.set_title("B. Matched-null ladder: hold each identity class "
                   "(and all three jointly) constant",
                   fontsize=10.5, fontweight="bold", loc="left")
    axB.legend(fontsize=8, loc="upper right", frameon=True)
    axB.tick_params(labelsize=8)
    for s in axB.spines.values(): s.set_color("#bbb")
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # ---- Panel C: projection triangulation ----
    axC.set_facecolor("white")
    if proj_mu is not None:
        labels = ["full", "−5 dir", "−10 dir", "−20 dir"]
        x = np.arange(len(labels))
        axC.bar(x, proj_mu, 0.6, color=["#222", "#b86a3f", "#cf8b5f", "#e0b090"],
                edgecolor="#222", lw=0.4)
        for i, v in enumerate(proj_mu):
            axC.text(i, v + 0.005, f"{v:+.2f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        axC.set_xticks(x); axC.set_xticklabels(labels, fontsize=9)
        axC.axhline(0, color="#666", lw=0.5)
        axC.set_ylabel(r"$\mu\Delta_{full}$ ($\times 10^{-3}$)", fontsize=9)
    else:
        axC.text(0.5, 0.5, "projection numbers\npending", ha="center",
                  va="center", transform=axC.transAxes, color="#999")
    axC.set_title("C. Species-subspace projection\n(remove species axis, recompute Δ_full)",
                   fontsize=10.5, fontweight="bold", loc="left")
    axC.tick_params(labelsize=8)
    for s in axC.spines.values(): s.set_color("#bbb")
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    pm = None
    if len(sys.argv) > 1:
        pm = [float(x) for x in sys.argv[1:]]
    main(pm)
