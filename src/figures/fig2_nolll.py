"""Fig 2 (method convergence): the biome-mythology alignment under TWO
independent strategies -- remove the names (LLM-strip, stratified/marginal
Delta) and hold the names constant (raw myths + discrete matched-permutation
null). A 2x2 grid (corpus x method); the same biomes survive under both.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
LAD = EMB / "ladder"
OUT = ROOT / "paper/figures/fig2_nolll.png"
from make_phase2_figures import short_biome, biome_color


def load():
    li = pd.read_csv(EMB / "inat_basic/v3_biome_test_sentpool_resid.csv")
    li = li[["biome", "delta_strat", "p_strat"]].rename(
        columns={"delta_strat": "delta", "p_strat": "p"})
    lp = pd.read_csv(EMB / "places365_strict/v3_biome_test_sentpool_resid.csv")
    lp = lp[["biome", "delta", "p_one_sided"]].rename(columns={"p_one_sided": "p"})
    mi = pd.read_csv(LAD / "nolll_perbiome_inat.csv")[
        ["biome", "delta_raw", "p_matched", "q_matched"]].rename(
        columns={"delta_raw": "delta", "p_matched": "p", "q_matched": "q"})
    mp = pd.read_csv(LAD / "nolll_perbiome_p365.csv")[
        ["biome", "delta_raw", "p_matched", "q_matched"]].rename(
        columns={"delta_raw": "delta", "p_matched": "p", "q_matched": "q"})
    return li, lp, mi, mp


def panel(ax, df, order, title, has_fdr):
    df = df.set_index("biome").reindex(order)
    y = np.arange(len(order))[::-1]
    vals = df["delta"].values * 1000
    ax.barh(y, np.nan_to_num(vals), color=[biome_color(b) for b in order],
            edgecolor="#222", lw=0.4, zorder=2)
    xmax = np.nanmax(np.abs(vals)) * 1.32 if np.isfinite(np.nanmax(vals)) else 1
    for i, b in enumerate(order):
        v = vals[i]
        if np.isnan(v):
            ax.text(0.01 * xmax, y[i], "n/a", va="center", fontsize=7, color="#aaa")
            continue
        sig = df["p"].values[i] < 0.05
        mk = ""
        if sig:
            mk = "$\\bigstar$" if (has_fdr and df["q"].values[i] < 0.05) else "$\\star$"
        if mk:
            x = v + 0.02 * xmax if v >= 0 else v - 0.02 * xmax
            ax.text(x, y[i], mk, va="center", ha="left" if v >= 0 else "right",
                    fontsize=11, color="#b8860b")
    ax.axvline(0, color="#666", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([short_biome(b) for b in order], fontsize=8)
    ax.set_xlim(min(0, np.nanmin(vals) * 1.2), xmax)
    n = int((~np.isnan(vals)).sum()); ns = int((df["p"] < 0.05).sum())
    ax.set_title(f"{title}  ·  {ns}/{n} at $p<.05$", fontsize=10.5,
                 fontweight="bold", loc="left")
    ax.set_xlabel(r"$\Delta$ ($\times 10^{-3}$)", fontsize=8.5)
    for s in ax.spines.values():
        s.set_color("#bbb")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main():
    li, lp, mi, mp = load()
    order = li.sort_values("delta", ascending=False)["biome"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    panel(axes[0, 0], li, order, "A  iNat · remove the names (LLM-strip, stratified)", False)
    panel(axes[0, 1], mi, order, "B  iNat · hold the names constant (raw + matched null)", True)
    panel(axes[1, 0], lp, order, "C  Places365 · remove the names (LLM-strip, marginal)", False)
    panel(axes[1, 1], mp, order, "D  Places365 · hold the names constant (raw + matched null)", True)

    # convergence numbers
    def agree(llm, mat):
        a = llm.set_index("biome")["p"] < 0.05
        b = mat.set_index("biome")["p"] < 0.05
        common = a.index.intersection(b.index)
        both = int((a[common] & b[common]).sum())
        m = pd.concat([llm.set_index("biome")["delta"], mat.set_index("biome")["delta"]],
                      axis=1).dropna()
        rho = stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1]).correlation
        return both, rho
    bi, ri = agree(li, mi); bp, rp = agree(lp, mp)
    fig.suptitle("Two roads to the same biomes: remove the names (left) and hold the names "
                 "constant (right) converge\n"
                 f"iNat: {bi} biomes significant under both, $\\Delta$ rank $\\rho={ri:.2f}$   ·   "
                 f"Places365: {bp} under both, $\\rho={rp:.2f}$   "
                 r"($\bigstar$ survives FDR $q<.05$, $\star$ nominal $p<.05$)",
                 fontsize=11.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"iNat agree={bi} rho={ri:.3f} | P365 agree={bp} rho={rp:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
