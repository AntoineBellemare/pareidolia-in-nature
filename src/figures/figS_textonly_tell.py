"""Text-only biome-tell split (reviewer A1): the alignment survives where a
purely text-side classifier cannot predict biome.

Compares two median splits of motifs by biome-predictability:
  - image-derived tell  (SigLIP similarity to per-biome image prototypes)
  - text-derived tell    (TF-IDF + logistic regression on the anonymised text,
                          out-of-fold; never sees an image)
and plots the within-iconic-taxon stratified muDelta on the low/high half of
each. The key point: the LOW half stays positive under both, and the
text-derived split is fully independent of the vision model.

Outputs paper/figures/figS_textonly_tell.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "dataset/imagery/embeddings/siglip2-large"
OUT = ROOT / "paper/figures/figS_textonly_tell.png"


def half_mu(csv, lo, hi):
    d = pd.read_csv(csv)
    m = d.groupby("half")["delta_strat"].mean() * 1000
    return float(m[lo]), float(m[hi])


def main():
    img_lo, img_hi = half_mu(EMB / "v3_biome_tell_split.csv", "low_tell", "high_tell")
    txt_lo, txt_hi = half_mu(EMB / "v3_biome_tell_textonly_split.csv",
                             "low_tell_textonly", "high_tell_textonly")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    groups = ["Image-derived tell\n(SigLIP prototypes)",
              "Text-derived tell\n(TF-IDF classifier, image-free)"]
    x = np.arange(2)
    wbar = 0.34
    lo = [img_lo, txt_lo]; hi = [img_hi, txt_hi]
    b1 = ax.bar(x - wbar / 2, lo, wbar, color="#6a9bd8", edgecolor="#222", lw=0.6,
                label="low-tell half (biome hard to predict)")
    b2 = ax.bar(x + wbar / 2, hi, wbar, color="#b8860b", edgecolor="#222", lw=0.6,
                label="high-tell half (biome easy to predict)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():+.2f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#222")
    ax.axhline(0.406, color="#c0392b", ls="--", lw=1.2,
               label="full-corpus headline (+0.41)")
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel(r"within-taxon stratified $\mu\Delta$ ($\times 10^{-3}$)", fontsize=10)
    ax.set_ylim(0, max(hi) * 1.25)
    ax.set_title("The biome alignment survives where biome is least predictable "
                 "from text\nLow-tell halves stay positive under both splits; the "
                 "text-derived split never sees an image",
                 fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(fontsize=8.6, loc="upper left", frameon=True)
    for s in ax.spines.values():
        s.set_color("#bbb")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"image tell: low {img_lo:+.3f} high {img_hi:+.3f}")
    print(f"text  tell: low {txt_lo:+.3f} high {txt_hi:+.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
