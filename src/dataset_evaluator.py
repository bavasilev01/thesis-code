#!/usr/bin/env python3
# static_dashboard.py

import os
import re
import logging
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mimic_dataset import MIMICDataset  # your existing loader

# ------------------------------------------------------------------------------
# Configuration & Logging
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
plt.style.use('classic')  # start with a clean, consistent base

OUTPUT_IMAGE = "static_dashboard.png"
FIGSIZE = (16, 24)  # width, height in inches

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def collect_texts(dataset):
    reports, structs = [], []
    for idx in range(len(dataset)):
        try:
            _, report, struct = dataset[idx]
            reports.append(report)
            structs.append(struct)
        except Exception as e:
            logging.warning(f"Skipping record {idx}: {e}")
    if not reports or not structs:
        raise ValueError("No text data could be collected.")
    return reports, structs

def text_length_stats(texts):
    lengths = np.array([len(tokenize(t)) for t in texts])
    return lengths, {
        'count': len(lengths),
        'min': int(lengths.min()),
        'max': int(lengths.max()),
        'mean': float(lengths.mean()),
        'median': float(np.median(lengths)),
        'std': float(lengths.std()),
    }

def top_n_words(texts, n=20):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    return counter.most_common(n)

def phrase_freqs(texts, phrases):
    freqs = {p:0 for p in phrases}
    for t in texts:
        for p in phrases:
            if p in t:
                freqs[p] += 1
    return freqs

# ------------------------------------------------------------------------------
# Main plotting routine
# ------------------------------------------------------------------------------

def create_static_dashboard(mimic: MIMICDataset, output_path=OUTPUT_IMAGE):
    # 1) load table & texts
    df = mimic.dataset_table
    label_cols = [c for c in df.columns
                  if c not in ('image_path','report_path','subject_id','study_id')]
    reports, structs = collect_texts(mimic)

    # 2) compute stats
    report_lengths, r_stats = text_length_stats(reports)
    struct_lengths, s_stats = text_length_stats(structs)
    vocab_size = len({w for t in reports for w in tokenize(t)})
    top_words = top_n_words(reports, n=20)
    phrase_list = ["Evidence of:", "No evidence of", "Uncertain regarding", "No significant findings."]
    phrase_stats = phrase_freqs(structs, phrase_list)

    # 3) label distributions
    lab = df[label_cols]
    total = lab.size
    counts = {
        'Positive (1)': int((lab==1.0).sum().sum()),
        'Negative (0)': int((lab==0.0).sum().sum()),
        'Uncertain (-1)': int((lab==-1.0).sum().sum()),
        'Missing': int(lab.isna().sum().sum())
    }
    per_label = lab.apply(lambda col: {
        'pos': int((col==1.0).sum()),
        'neg': int((col==0.0).sum()),
        'unc': int((col==-1.0).sum()),
        'mis': int(col.isna().sum())
    }, axis=0)
    # correlation matrix (fill NaN→0 so that missingness doesn’t yield NaN corr)
    corr = lab.fillna(0).corr()

    # 4) build figure
    fig = plt.figure(figsize=FIGSIZE)
    gs = gridspec.GridSpec(5, 2, height_ratios=[1,1,1,1,1], hspace=0.4, wspace=0.3)

    # -- Row 1: Report length hist & structured length hist --
    ax0 = fig.add_subplot(gs[0,0])
    ax0.hist(report_lengths, bins=50, alpha=0.8)
    ax0.set_title("Report Length Distribution (words)")
    ax0.set_xlabel("Word Count")
    ax0.set_ylabel("Frequency")
    ax0.grid(True, linestyle='--', alpha=0.5)

    ax1 = fig.add_subplot(gs[0,1])
    ax1.hist(struct_lengths, bins=30, color='C1', alpha=0.8)
    ax1.set_title("Structured‐Text Length Distribution")
    ax1.set_xlabel("Word Count")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # -- Row 2: Top words & phrase freqs --
    ax2 = fig.add_subplot(gs[1,0])
    words, wcounts = zip(*top_words)
    ax2.barh(words[::-1], wcounts[::-1], color='C2')
    ax2.set_title(f"Top 20 Words in Reports (vocab={vocab_size})")
    ax2.set_xlabel("Count")
    for i, v in enumerate(wcounts[::-1]):
        ax2.text(v + max(wcounts)*0.01, i, str(v), va='center')
    ax2.grid(axis='x', linestyle='--', alpha=0.4)

    ax3 = fig.add_subplot(gs[1,1])
    phrases, pcounts = zip(*phrase_stats.items())
    ax3.bar(phrases, pcounts, color='C3')
    ax3.set_title("Key Phrase Frequency in Structured Texts")
    ax3.set_xticklabels(phrases, rotation=45, ha='right')
    ax3.set_ylabel("Count")
    for idx, val in enumerate(pcounts):
        ax3.text(idx, val+0.5, str(val), ha='center')
    ax3.grid(axis='y', linestyle='--', alpha=0.4)

    # -- Row 3: Overall label distribution & per‐label missingness bar --
    ax4 = fig.add_subplot(gs[2,0])
    ax4.pie(counts.values(), labels=counts.keys(),
            autopct='%1.1f%%', startangle=90, wedgeprops={'width':0.4})
    ax4.set_title("Overall Label Distribution")

    ax5 = fig.add_subplot(gs[2,1])
    misrates = {lbl: v['mis'] / (v['pos']+v['neg']+v['unc']+v['mis'])
                for lbl, v in per_label.items()}
    # sort by missingness
    lbls, rates = zip(*sorted(misrates.items(), key=lambda x: x[1], reverse=True))
    ax5.barh(lbls[::-1], [r*100 for r in rates[::-1]], color='C4')
    ax5.set_title("Per‐Label Missingness (%)")
    ax5.set_xlabel("% Missing")
    ax5.grid(axis='x', linestyle='--', alpha=0.5)

    # -- Row 4: Label‐correlation heatmap --
    ax6 = fig.add_subplot(gs[3,:])
    im = ax6.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax6.set_title("Inter-Label Correlation Matrix")
    ax6.set_xticks(np.arange(len(label_cols)))
    ax6.set_yticks(np.arange(len(label_cols)))
    ax6.set_xticklabels(label_cols, rotation=90, fontsize=8)
    ax6.set_yticklabels(label_cols, fontsize=8)
    plt.colorbar(im, ax=ax6, fraction=0.02, pad=0.01)

    # -- Row 5: Report vs. structured length scatter --
    ax7 = fig.add_subplot(gs[4,:])
    ax7.scatter(report_lengths, struct_lengths, alpha=0.3, s=10)
    ax7.set_title("Report Length vs Structured-Text Length")
    ax7.set_xlabel("Report Length (words)")
    ax7.set_ylabel("Structured Length (words)")
    ax7.grid(True, linestyle='--', alpha=0.5)

    # -- Figure‐level title & save --
    fig.suptitle("MIMIC Dataset Static EDA Dashboard", fontsize=24, weight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ensure output dir exists
    outdir = os.path.dirname(output_path) or '.'
    os.makedirs(outdir, exist_ok=True)

    fig.savefig(output_path, dpi=150)
    logging.info(f"Saved dashboard to {output_path}")

# ------------------------------------------------------------------------------
# CLI entry‐point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a static EDA dashboard for a MIMICDataset."
    )
    parser.add_argument(
        "--csv", "-c", required=True,
        help="Path to your dataset CSV (used by MIMICDataset)."
    )
    parser.add_argument(
        "--out", "-o", default=OUTPUT_IMAGE,
        help="Output file for the dashboard (png or pdf)."
    )
    args = parser.parse_args()

    logging.info(f"Loading MIMICDataset from {args.csv!r} …")
    ds = MIMICDataset(dataset_csv=args.csv)
    logging.info(f"Dataset contains {len(ds)} samples.")
    create_static_dashboard(ds, output_path=args.out)
