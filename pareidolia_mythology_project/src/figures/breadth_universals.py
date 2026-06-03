"""FIG 11 — Universals-only analysis.

Universal motifs = motifs touching ≥4 biomes ("widespread mythology").
For each biome, compute residualised Δ between universal motifs that touch
that biome vs universal motifs that don't.

If universals have NO biome-specific alignment → validates the Spec A
approach. If universals DO show alignment → suggests deeper effect that
extends to widespread mythology too.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # project root
MAP = ROOT / "dataset/mapping_v2"
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
OUT = ROOT / "dataset/imagery/figures/headlines_final_russian/fig11_universals_analysis.png"

from make_phase2_figures import short_biome, biome_color, sig_stars


def bh(p):
    p = np.asarray(p, float); valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if not valid.any(): return q
    pv = p[valid]; n = len(pv)
    order = np.argsort(pv); ranked = pv[order]
    qv = ranked * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    unranked = np.empty_like(qv); unranked[order] = qv
    q[valid] = unranked; return q


def run_breadth_split(img_emb, img_meta, motif_emb, motif_meta,
                       motif_to_biomes, breadth_label,
                       breadth_predicate, n_perms=1000,
                       motif_biome_owntrads=None,
                       owntrad_threshold=3):
    """For each biome, residualised Δ between in-B vs out-of-B motifs
    that satisfy `breadth_predicate(n_biomes)`.

    If `motif_biome_owntrads` is provided (dict[motif_id][biome] = #trads),
    we also require own_trads(motif, biome) >= `owntrad_threshold` for the
    motif to count as "in biome B" — i.e. Spec A semantics. This is the
    extra condition that makes the LEFT panel match Fig 2.
    """
    sims = img_emb @ motif_emb.T
    sims = sims - sims.mean(axis=0, keepdims=True)
    motif_ids = motif_meta["motif_id"].tolist()
    biomes = sorted({b for s in motif_to_biomes.values() for b in s})

    keep_mask = np.array([
        breadth_predicate(len(motif_to_biomes.get(mid, set())))
        for mid in motif_ids
    ])
    is_in_biome = np.array([
        [b in motif_to_biomes.get(mid, set()) for b in biomes]
        for mid in motif_ids
    ])
    if motif_biome_owntrads is not None:
        owntrad_ok = np.array([
            [motif_biome_owntrads.get(mid, {}).get(b, 0) >= owntrad_threshold
             for b in biomes]
            for mid in motif_ids
        ])
        is_in_biome = is_in_biome & owntrad_ok
    use_biome = img_meta.get("photo_biome_wwf")
    if use_biome is None: use_biome = img_meta["tradition_biome_wwf"]
    use_biome = use_biome.fillna(img_meta.get("tradition_biome_wwf", "")).values

    rng = np.random.default_rng(42)
    rows = []
    for j, b in enumerate(biomes):
        b_imgs = use_biome == b
        if b_imgs.sum() < 5: continue
        in_b = is_in_biome[:, j] & keep_mask
        out_b = (~is_in_biome[:, j]) & keep_mask
        if in_b.sum() < 5 or out_b.sum() < 5: continue
        per = sims[b_imgs].mean(axis=0)
        delta = float(per[in_b].mean() - per[out_b].mean())

        # Permutation null
        keep_idx = np.where(keep_mask)[0]
        in_b_keep = is_in_biome[keep_idx, j]
        null = np.empty(n_perms)
        for k in range(n_perms):
            shuf = rng.permutation(in_b_keep)
            null[k] = per[keep_idx[shuf]].mean() - per[keep_idx[~shuf]].mean()
        p = float((null >= delta).mean())
        rows.append({
            "biome": b, "n_imgs": int(b_imgs.sum()),
            "n_motifs_in_biome": int(in_b.sum()),
            "n_motifs_out": int(out_b.sum()),
            "delta": delta, "p_one_sided": p,
            "breadth": breadth_label,
        })
    return pd.DataFrame(rows)


MIN_MOTIFS = 10  # match fig2's display threshold


def _panel(ax, df, title, has_y=True, apply_min_motifs=True,
            common_order=None, xmax_global=None):
    """Print-style horizontal bar panel, white background, biome-color tick swatches."""
    ax.set_facecolor("white")
    df = df.copy()
    df = df[df["biome"].apply(lambda x: isinstance(x, str)) & (df["biome"] != "N/A")]
    if apply_min_motifs:
        df = df[df["n_motifs_in_biome"] >= MIN_MOTIFS]
        if df.empty:
            ax.set_title(f"{title}\n(no cells ≥{MIN_MOTIFS} motifs)", color="#888")
            return
    df["q"] = bh(df["p_one_sided"].values)
    if common_order is not None:
        df = df.set_index("biome").reindex(common_order).reset_index()
        df = df.dropna(subset=["delta"])
    else:
        df = df.sort_values("delta")
    y = np.arange(len(df))
    colors = [biome_color(b) for b in df["biome"]]

    # Soft grid + zero line
    ax.axvline(0, color="#777", lw=0.8, zorder=1)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cccccc", zorder=0)

    for yi, c, d in zip(y, colors, df["delta"]):
        ax.barh(yi, d, color=c, edgecolor=c, lw=0.8, alpha=0.92, height=0.72,
                zorder=2)

    if xmax_global is None:
        xmax = max(abs(df["delta"]).max() * 1.55, 1e-7)
    else:
        xmax = xmax_global * 1.55
    for i, (_, r) in enumerate(df.reset_index(drop=True).iterrows()):
        s = sig_stars(r["p_one_sided"])
        fdr = (not pd.isna(r["q"])) and (r["q"] < 0.05)
        x = r["delta"]
        label_x = x + xmax*0.012 if x >= 0 else x - xmax*0.012
        ha = "left" if x >= 0 else "right"
        d_label = f"{r['delta']*1000:+.2f}"
        if s:
            d_label = f"{d_label} {s}"
        ax.text(label_x, i, d_label,
                color=("#a36b00" if fdr else "#444"),
                fontsize=9.0,
                fontweight=("bold" if fdr else "normal"),
                va="center", ha=ha, zorder=3)
        ax.text(xmax * 1.02, i,
                f"n_motif={int(r['n_motifs_in_biome'])}",
                color="#666", fontsize=7.2, va="center", ha="left", zorder=3)
    ax.set_xlim(-xmax, xmax * 1.55)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_yticks(y)
    if has_y:
        ax.set_yticklabels([short_biome(b) for b in df["biome"]],
                            color="#222", fontsize=9.5)
        for tick, b in zip(ax.get_yticklabels(), df["biome"]):
            tick.set_bbox(dict(facecolor=biome_color(b), edgecolor="none",
                                pad=2.5, alpha=0.18))
    else:
        ax.set_yticklabels([])
    n_sig = int((df["p_one_sided"] < 0.05).sum())
    n_fdr = int((df["q"] < 0.05).sum())
    ax.set_title(f"{title}\nμΔ={df['delta'].mean()*1000:+.2f}×10⁻³  ·  "
                 f"sig={n_sig}/{len(df)}  ·  FDR={n_fdr}/{len(df)}",
                 color="#111", fontsize=10.5, fontweight="bold",
                 loc="left", pad=8)
    ax.set_xlabel(r"$\Delta$ ($\times 10^{-3}$ at labels)",
                  color="#444", fontsize=8.5)
    ax.tick_params(colors="#444", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#aaa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("Loading …")
    img_emb = np.load(EMB / "inat_basic/img_emb.npy")
    img_meta = pd.read_parquet(EMB / "inat_basic/img_paths.parquet").reset_index(drop=True)
    # NOTE: keep v8 here because the universals analysis needs ALL motifs
    # (Spec A + semi-universal + universal). The Gemini LLM-clean text
    # is only available for the 547 Spec A motifs; semi-universal and
    # universal are still on v8. This figure compares breadth strata
    # under the v8 anonymisation; the LLM-clean strata comparison will
    # come when the extension Gemini run on non-Spec-A motifs lands.
    motif_emb = np.load(EMB / "motif_emb_abstracts_ru_hypv8.npy")
    motif_meta = pd.read_parquet(EMB / "motif_meta_abstracts_ru_hypv8.parquet")

    # Apply biome-tell filter to motifs
    filt = pd.read_csv(EMB / "biome_tell_filter_v8.csv")
    keep = set(filt.query("keep_in_filtered")["motif_id"].astype(str))
    mask = motif_meta["motif_id"].astype(str).isin(keep).values
    motif_meta = motif_meta[mask].reset_index(drop=True)
    motif_emb = motif_emb[mask]

    # Motif → biome membership
    tm = pd.read_parquet(MAP / "tradition_motif.parquet")
    trad = pd.read_parquet(MAP / "traditions.parquet")
    tm = tm.copy()
    tm["biome_wwf"] = tm["oid"].map(trad.set_index("oid")["biome_wwf"])
    motif_to_biomes = (tm.groupby("motif_id")["biome_wwf"]
                       .apply(lambda s: set(b for b in s if isinstance(b, str)))
                       .to_dict())
    # Motif × biome → own-tradition count (for Spec A on the LEFT panel)
    cnt = (tm.groupby(["motif_id", "biome_wwf"]).size()
             .unstack(fill_value=0))
    motif_biome_owntrads = {mid: dict(row.items()) for mid, row in cnt.iterrows()}

    # Distribution of motif breadth
    print("\nBreadth distribution among filtered motifs:")
    breadths = np.array([
        len(motif_to_biomes.get(mid, set())) for mid in motif_meta["motif_id"]
    ])
    bins = [(1, 3, "biome-specific (1-3 biomes)"),
            (4, 7, "semi-universal (4-7 biomes)"),
            (8, 14, "universal (8+ biomes)")]
    for lo, hi, label in bins:
        n = ((breadths >= lo) & (breadths <= hi)).sum()
        print(f"  {label:35s}: {n} motifs")

    # Run three panels
    results = []
    print("\nComputing per-breadth-bin Δ per biome …")
    # LEFT panel = Spec A: ≤3 biomes AND ≥3 own-trads per biome → MATCHES FIG 2
    df_spec = run_breadth_split(img_emb, img_meta, motif_emb, motif_meta,
                                 motif_to_biomes,
                                 "Spec A (≤3 biomes, ≥3 own-trads per biome)",
                                 lambda k: 1 <= k <= 3,
                                 motif_biome_owntrads=motif_biome_owntrads,
                                 owntrad_threshold=3)
    # Middle/right panels = breadth only (universals don't usefully gate on own-trads,
    # since their many-tradition footprint already covers each biome ≥3 times)
    df_semi = run_breadth_split(img_emb, img_meta, motif_emb, motif_meta,
                                 motif_to_biomes, "semi-universal (4-7 biomes)",
                                 lambda k: 4 <= k <= 7)
    df_univ = run_breadth_split(img_emb, img_meta, motif_emb, motif_meta,
                                 motif_to_biomes, "universal (8+ biomes)",
                                 lambda k: k >= 8)

    # Common biome order: ascending iNat Δ of the Spec A panel (matches fig 2 logic)
    spec_for_order = df_spec.copy()
    spec_for_order = spec_for_order[
        spec_for_order["biome"].apply(lambda x: isinstance(x, str))
        & (spec_for_order["biome"] != "N/A")]
    spec_for_order = spec_for_order[
        spec_for_order["n_motifs_in_biome"] >= MIN_MOTIFS]
    common_order = spec_for_order.sort_values("delta")["biome"].tolist()
    xmax_global = max(
        abs(df_spec["delta"]).max(),
        abs(df_semi["delta"]).max(),
        abs(df_univ["delta"]).max(),
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.patch.set_facecolor("white")
    _panel(axes[0], df_spec,
           "Biome-specific (Spec A: ≤3 biomes, ≥3 own-trads)\n[same selection as fig 2]",
           has_y=True, common_order=common_order, xmax_global=xmax_global)
    _panel(axes[1], df_semi,
           "Semi-universal (4–7 biomes)\n[intermediate breadth]",
           has_y=True, common_order=common_order, xmax_global=xmax_global)
    _panel(axes[2], df_univ,
           "Universal (≥8 biomes)\n[widely-shared mythology]",
           has_y=True, common_order=common_order, xmax_global=xmax_global)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
