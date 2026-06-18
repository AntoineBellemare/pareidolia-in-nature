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
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18],
                          width_ratios=[1.3, 0.78], hspace=0.62, wspace=0.32)
    fig.patch.set_facecolor("white")
    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, :])

    # ---- Panel A: channel summary (muDelta with 95% CI, per channel) ----
    axA.set_facecolor("white")
    chan = pd.read_csv(LAD / "stats_channel_summary_strat.csv").set_index("channel")
    cols = {"full": "#222222", "species": "#cf6f3f",
            "place": "#5b8db8", "ethnonym": "#7cbe5e"}
    nice = {"full": "full original myth", "species": "species bag",
            "ethnonym": "ethnonym bag", "place": "place bag"}
    seqA = ["species", "ethnonym", "full", "place"]          # descending muDelta
    yA = np.arange(len(seqA))[::-1]
    for i, ch in enumerate(seqA):
        r = chan.loc[ch]
        err = [[r["mu"] - r["ci_lo"]], [r["ci_hi"] - r["mu"]]]
        axA.barh(yA[i], r["mu"], height=0.62, color=cols[ch], edgecolor="#222",
                 lw=0.5, xerr=err, error_kw=dict(ecolor="#555", lw=1.3, capsize=3),
                 zorder=2)
        axA.text(r["ci_hi"] + 0.04, yA[i], f"{int(r['n_sig'])}/{int(r['n'])} sig",
                 va="center", ha="left", fontsize=8.5, color="#333")
    axA.set_yticks(yA); axA.set_yticklabels([nice[c] for c in seqA], fontsize=9.5)
    axA.set_xlim(0, 1.55)
    axA.set_xlabel(r"stratified $\mu\Delta$ ($\times 10^{-3}$, raw-Russian frame)",
                   fontsize=9.5)
    axA.set_title("A  Each channel is biome-diagnostic\n"
                  r"species bag $>$ full myth",
                  fontsize=10.5, fontweight="bold", loc="left")
    for s in axA.spines.values(): s.set_color("#bbb")
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

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

    # ---- Panel B: which biomes stay significant after holding identity constant ----
    axB.set_facecolor("white")
    nulls = [("species", "Species"), ("ethnonym", "Ethnonym"),
             ("place", "Place"), ("joint", "Joint  (species+place+ethnonym)")]
    data = {k: pd.read_csv(LAD / f"stats_{k}_matched_null_strat.csv").set_index("biome")
            for k, _ in nulls}
    biomes = list(data["joint"].index)
    nsurv = {b: int(sum(float(data[k].loc[b, "q"]) < 0.05 for k, _ in nulls))
             for b in biomes}
    biomes.sort(key=lambda b: (-nsurv[b], -float(data["joint"].loc[b, "delta_obs"])))
    yrows = {"species": 3, "ethnonym": 2, "place": 1, "joint": 0}
    for k, _ in nulls:
        d = data[k]
        for xi, b in enumerate(biomes):
            r = d.loc[b]
            surv = float(r["q"]) < 0.05            # FDR q<.05
            dd = max(float(r["delta_obs"]) * 1000, 0.0)
            size = 45 + 470 * min(dd, 1.1) / 1.1
            if surv:
                axB.scatter(xi, yrows[k], s=size, c=[biome_color(b)],
                            edgecolors="#111", linewidths=0.9, zorder=3)
            else:
                axB.scatter(xi, yrows[k], s=size, facecolors="none",
                            edgecolors="#c8c8c8", linewidths=1.1, zorder=2)
    axB.axhline(0.5, color="#888", lw=0.9, ls="--", zorder=1)   # set off the joint row
    cnt = {k: int((data[k]["q"] < 0.05).sum()) for k, _ in nulls}
    axB.set_yticks([yrows[k] for k, _ in nulls])
    axB.set_yticklabels([f"{lab}\n{cnt[k]}/14 survive" for k, lab in nulls], fontsize=9)
    axB.set_xticks(range(len(biomes)))
    axB.set_xticklabels([short_biome(b) for b in biomes], rotation=38, ha="right",
                        fontsize=8.5)
    axB.set_ylim(-0.7, 3.7); axB.set_xlim(-0.7, len(biomes) - 0.3)
    axB.set_title("B  Which biomes stay significant after holding identity constant\n"
                  r"filled $=$ survives the matched null at FDR $q<.05$; "
                  r"marker size $\propto$ observed $\Delta$",
                  fontsize=10.5, fontweight="bold", loc="left")
    for s in axB.spines.values(): s.set_visible(False)
    axB.tick_params(length=0)

    fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    pm = None
    if len(sys.argv) > 1:
        pm = [float(x) for x in sys.argv[1:]]
    main(pm)
