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
OUT = ROOT / "paper/figures/fig_identity_naming.png"

from make_phase2_figures import short_biome, biome_color

# projection μΔ (×10⁻³) for [base, 5, 10, 20] components removed — filled from stats stdout
PROJ = {"labels": ["0 (full)", "−5 dir", "−10 dir", "−20 dir"],
        "mu": None}  # set below if available


def main(proj_mu=None):
    fig = plt.figure(figsize=(16, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.05, 0.9], wspace=0.34)
    fig.patch.set_facecolor("white")
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])

    # ---- Panel A: channel summary (muDelta with 95% CI, per channel) ----
    axA.set_facecolor("white")
    chan = pd.read_csv(LAD / "stats_channel_summary_strat.csv").set_index("channel")
    cols = {"full": "#222222", "species": "#cf6f3f",
            "place": "#5b8db8", "ethnonym": "#7cbe5e"}
    nice = {"full": "full original myth", "species": "species bag",
            "ethnonym": "ethnonym bag", "place": "place bag"}
    seqA = ["species", "ethnonym", "full", "place"]          # descending muDelta
    yA = np.arange(len(seqA))[::-1]
    full_mu = float(chan.loc["full", "mu"])
    for i, ch in enumerate(seqA):
        r = chan.loc[ch]
        err = [[r["mu"] - r["ci_lo"]], [r["ci_hi"] - r["mu"]]]
        axA.barh(yA[i], r["mu"], height=0.62, color=cols[ch], edgecolor="#222",
                 lw=0.5, xerr=err, error_kw=dict(ecolor="#555", lw=1.3, capsize=3),
                 zorder=2)
        axA.text(r["ci_hi"] + 0.04, yA[i], f"{int(r['n_sig'])}/{int(r['n'])} sig",
                 va="center", ha="left", fontsize=8.5, color="#333")
    axA.axvline(full_mu, ls="--", color="#222", lw=1.0, alpha=0.45, zorder=1)
    axA.set_yticks(yA); axA.set_yticklabels([nice[c] for c in seqA], fontsize=9.5)
    axA.set_xlim(0, 1.55)
    axA.set_xlabel(r"stratified $\mu\Delta$ ($\times 10^{-3}$, raw-Russian frame)",
                   fontsize=9.5)
    axA.set_title("A  Channels are biome-diagnostic\n"
                  r"species bag $>$ full myth",
                  fontsize=10.5, fontweight="bold", loc="left")
    for s in axA.spines.values(): s.set_color("#bbb")
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # ---- Panel B: matched-null ladder ----
    axB.set_facecolor("white")
    summ = pd.read_csv(LAD / "stats_matched_null_summary_strat.csv")
    o = {r["null"]: r for _, r in summ.iterrows()}
    seq = [s for s in ["species", "ethnonym", "place", "joint"] if s in o]
    x = np.arange(len(seq)); bw = 0.38
    obs = [o[s]["obs"] for s in seq]
    nul = [o[s]["null_mean"] for s in seq]
    axB.bar(x - bw/2, obs, bw, color="#222222", edgecolor="#111", lw=0.4,
            label=r"observed $\Delta_{\mathrm{full}}$", zorder=2)
    axB.bar(x + bw/2, nul, bw, color="#c9a04a", edgecolor="#111", lw=0.4,
            label="matched-null mean", zorder=2)
    for i, s in enumerate(seq):
        axB.text(i, max(obs[i], nul[i]) + 0.015,
                 f"{int(o[s]['n_sig'])}/{int(o[s]['n'])}", ha="center",
                 va="bottom", fontsize=9, fontweight="bold", color="#222")
    axB.set_xticks(x)
    axB.set_xticklabels([s.capitalize() for s in seq], fontsize=9.5)
    axB.set_ylabel(r"$\mu\Delta$ ($\times 10^{-3}$)", fontsize=9.5)
    axB.set_ylim(0, max(obs) * 1.45)
    axB.set_title("B  Holding identity constant\n"
                  r"biomes surviving the null ($p<.05$)",
                  fontsize=10.5, fontweight="bold", loc="left")
    axB.legend(fontsize=8.5, loc="upper right", frameon=True)
    for s in axB.spines.values(): s.set_color("#bbb")
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # ---- Panel C: species-subspace projection ----
    axC.set_facecolor("white")
    if proj_mu is not None:
        labels = ["full", "$-5$", "$-10$", "$-20$"]
        x = np.arange(len(labels))
        axC.bar(x, proj_mu, 0.62, color=["#222", "#b86a3f", "#cf8b5f", "#e0b090"],
                edgecolor="#222", lw=0.4, zorder=2)
        for i, v in enumerate(proj_mu):
            axC.text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=9.5, fontweight="bold")
        axC.set_xticks(x); axC.set_xticklabels(labels, fontsize=9.5)
        axC.set_xlabel("species directions removed", fontsize=9)
        axC.axhline(0, color="#666", lw=0.5)
        axC.set_ylim(0, max(proj_mu) * 1.2)
        axC.set_ylabel(r"$\mu\Delta_{\mathrm{full}}$ ($\times 10^{-3}$)", fontsize=9.5)
    axC.set_title("C  Removing the species axis\n"
                  "attenuates but persists",
                  fontsize=10.5, fontweight="bold", loc="left")
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
