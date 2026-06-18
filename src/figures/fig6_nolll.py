"""Fig 6 (method convergence): per-(biome x taxon) alignment under both
strategies. Left = remove the names (LLM-strip, stratified Delta, perm null);
right = hold the names constant (raw myths, discrete matched null, BH-FDR).
Stars mark significant cells; the same cells light up under both.
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
OUT = ROOT / "paper/figures/fig6_nolll.png"
from make_phase2_figures import short_biome

TAXA = ["Mammalia", "Aves", "Reptilia", "Amphibia", "Actinopterygii",
        "Insecta", "Arachnida", "Mollusca", "Plantae", "Fungi"]


def grid(df, biomes, tcol, dcol, sigmask):
    D = np.full((len(biomes), len(TAXA)), np.nan)
    S = np.zeros((len(biomes), len(TAXA)), bool)
    bi = {b: i for i, b in enumerate(biomes)}
    ti = {t: j for j, t in enumerate(TAXA)}
    for _, r in df.iterrows():
        if r["biome"] in bi and r[tcol] in ti:
            D[bi[r["biome"]], ti[r[tcol]]] = r[dcol]
            S[bi[r["biome"]], ti[r[tcol]]] = bool(sigmask.loc[r.name])
    return D, S


def panel(ax, D, S, biomes, title, fig, nsig, ntot):
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(D), cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(len(biomes)):
        for j in range(len(TAXA)):
            if S[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=12,
                        color="#111", fontweight="bold")
    ax.set_xticks(range(len(TAXA)))
    ax.set_xticklabels(TAXA, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(biomes)))
    ax.set_yticklabels([short_biome(b) for b in biomes], fontsize=8)
    ax.set_title(f"{title}\n* significant cells: {nsig}/{ntot}", fontsize=10.5,
                 fontweight="bold", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$\Delta$ ($\times 10^{-3}$)", fontsize=8)


def main():
    pb = pd.read_csv(LAD / "nolll_perbiome_inat.csv")
    biomes = pb.sort_values("delta_raw", ascending=False)["biome"].tolist()

    llm = pd.read_csv(EMB / "v3_byTaxon_sentpool_iNat.csv")
    llm = llm[llm["taxon_group"] != "all"].copy()
    llm["d"] = llm["delta"] * 1000
    Dl, Sl = grid(llm, biomes, "taxon_group", "d", llm["p_one_sided"] < 0.05)

    mat = pd.read_csv(LAD / "nolll_pertaxon_raw.csv")
    Dm, Sm = grid(mat, biomes, "taxon", "delta", mat["p"] < 0.05)   # nominal, matches panel A

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6.4))
    fig.patch.set_facecolor("white")
    panel(axA, Dl, Sl, biomes, "A  Remove the names (LLM-strip, perm null)",
          fig, int(Sl.sum()), int(np.isfinite(Dl).sum()))
    nq = int((mat["q"] < 0.05).sum())
    panel(axB, Dm, Sm, biomes,
          f"B  Hold the names constant (raw myths, matched null; {nq} survive FDR)",
          fig, int(Sm.sum()), int(np.isfinite(Dm).sum()))
    fig.suptitle("Per-taxon alignment converges across methods "
                 "(* marks significant biome-taxon cells)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
