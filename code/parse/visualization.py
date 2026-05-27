"""
Publication-quality visualization for PARSE CIT data.

Generates figures for the paper: CIT heatmaps, correlation matrices,
capability cliff quantification, and per-profile analysis.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Style ────────────────────────────────────────────────────────────

PAPER_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

COLORS = {
    "lang": "#2E86AB",     # teal
    "disc": "#A23B72",     # magenta
    "scen": "#F18F01",     # amber
    "deep": "#D64045",     # red
    "shallow": "#577399",  # slate
    "heatmap": "YlOrRd",
    "correlation": "coolwarm",
}


def apply_style():
    """Apply PARSE paper matplotlib style."""
    for k, v in PAPER_STYLE.items():
        plt.rcParams[k] = v


# ── Figure 1: Tri-axial CIT Heatmap ──────────────────────────────────

def figure_cit_heatmap(
    cit_lang: np.ndarray, lang_cats: List[str],
    cit_disc: np.ndarray, disc_cats: List[str],
    cit_scen: np.ndarray, scen_cats: List[str],
    output_path: str,
    cliff_ratios: Optional[Dict[str, float]] = None,
):
    """Generate Figure 1: CIT heatmap + correlation matrix + cliff bars."""
    apply_style()
    all_cats = lang_cats + disc_cats + scen_cats
    cit_combined = np.hstack([cit_lang, cit_disc, cit_scen])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={'width_ratios': [1, 0.7, 0.5]})

    # (a) CIT Heatmap
    im = axes[0].imshow(cit_combined.T, aspect='auto', cmap=COLORS["heatmap"],
                        interpolation='nearest')
    axes[0].set_yticks(range(len(all_cats)))
    axes[0].set_yticklabels(all_cats, fontsize=7)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xticklabels(range(0, 24, 2))
    axes[0].set_xlabel("Layer Index", fontweight='bold')
    axes[0].set_ylabel("Capability Axis", fontweight='bold')
    axes[0].set_title("(a) CIT Heatmap", fontweight='bold')
    axes[0].axvspan(0, 5.5, alpha=0.08, color='blue', label='Shallow (0-5)')
    axes[0].axvspan(15.5, 23, alpha=0.08, color='red', label='Deep (16-23)')
    axes[0].legend(fontsize=8, loc='upper left')
    plt.colorbar(im, ax=axes[0], shrink=0.8, label='Normalized CIT Score')

    # (b) Correlation matrix
    all_vecs = cit_combined
    n = all_vecs.shape[1]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(all_vecs[:, i], all_vecs[:, j])[0, 1]

    im2 = axes[1].imshow(corr, aspect='auto', cmap=COLORS["correlation"],
                         vmin=0.97, vmax=1.0, interpolation='nearest')
    axes[1].set_xticks(range(len(all_cats)))
    axes[1].set_xticklabels(all_cats, fontsize=6, rotation=45, ha='right')
    axes[1].set_yticks(range(len(all_cats)))
    axes[1].set_yticklabels(all_cats, fontsize=6)
    axes[1].set_title("(b) Cross-Axis Pearson $r$", fontweight='bold')

    # Block outlines
    n_lang, n_disc = len(lang_cats), len(disc_cats)
    axes[1].add_patch(plt.Rectangle((-0.5, -0.5), n_lang, n_lang, fill=False,
                                    edgecolor='green', linewidth=2, linestyle='--'))
    axes[1].text(n_lang/2 - 0.5, -1.2, 'Lang', fontsize=8, ha='center', color='green')
    axes[1].add_patch(plt.Rectangle((n_lang-0.5, n_lang-0.5), n_disc, n_disc, fill=False,
                                    edgecolor='blue', linewidth=2, linestyle='--'))
    axes[1].text(n_lang + n_disc/2 - 0.5, n_lang + 0.7, 'Disc', fontsize=8, ha='center', color='blue')
    n_scen = len(scen_cats)
    axes[1].add_patch(plt.Rectangle((n_lang+n_disc-0.5, n_lang+n_disc-0.5), n_scen, n_scen,
                                    fill=False, edgecolor='orange', linewidth=2, linestyle='--'))
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Pearson $r$')

    # (c) Capability Cliff
    if cliff_ratios is None:
        cliff_ratios = {}
        for name, cit in [("Language", cit_lang), ("Discipline", cit_disc), ("Scenario", cit_scen)]:
            r = cit[16:].mean(axis=0) / (cit[:6].mean(axis=0) + 1e-8)
            cliff_ratios[name] = float(r.mean())

    categories = list(cliff_ratios.keys())
    means = list(cliff_ratios.values())
    colors = [COLORS["lang"], COLORS["disc"], COLORS["scen"]]
    bars = axes[2].bar(categories, means, 0.5, color=colors, edgecolor='black', linewidth=0.5)
    for bar, mean_val in zip(bars, means):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{mean_val:.2f}×', ha='center', fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Deep/Shallow CIT Ratio", fontweight='bold')
    axes[2].set_title("(c) Capability Cliff", fontweight='bold')
    axes[2].set_ylim(0, max(means) * 1.3)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


# ── Figure 2: Detailed Correlation Matrices ──────────────────────────

def figure_correlation_matrices(
    cit_lang: np.ndarray, lang_cats: List[str],
    cit_disc: np.ndarray, disc_cats: List[str],
    output_path: str,
):
    """Generate Figure 2: Language-Language, Discipline-Discipline, and cross-axis correlations."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Language-Language
    n_lang = len(lang_cats)
    lang_corr = np.zeros((n_lang, n_lang))
    for i in range(n_lang):
        for j in range(n_lang):
            lang_corr[i, j] = np.corrcoef(cit_lang[:, i], cit_lang[:, j])[0, 1]
    r_lang = float(np.mean(lang_corr[np.triu_indices(n_lang, 1)]))

    im_a = axes[0].imshow(lang_corr, cmap=COLORS["correlation"], vmin=0.97, vmax=1.0)
    axes[0].set_xticks(range(n_lang)); axes[0].set_xticklabels(lang_cats, fontsize=10)
    axes[0].set_yticks(range(n_lang)); axes[0].set_yticklabels(lang_cats, fontsize=10)
    axes[0].set_title(f"(a) Language-Language\n$\\bar{{r}} = {r_lang:.4f}$", fontweight='bold')
    plt.colorbar(im_a, ax=axes[0], shrink=0.8)

    # (b) Discipline-Discipline
    n_disc = len(disc_cats)
    disc_corr = np.zeros((n_disc, n_disc))
    for i in range(n_disc):
        for j in range(n_disc):
            disc_corr[i, j] = np.corrcoef(cit_disc[:, i], cit_disc[:, j])[0, 1]
    r_disc = float(np.mean(disc_corr[np.triu_indices(n_disc, 1)]))

    im_b = axes[1].imshow(disc_corr, cmap=COLORS["correlation"], vmin=0.97, vmax=1.0)
    axes[1].set_xticks(range(n_disc)); axes[1].set_xticklabels(disc_cats, fontsize=10)
    axes[1].set_yticks(range(n_disc)); axes[1].set_yticklabels(disc_cats, fontsize=10)
    axes[1].set_title(f"(b) Discipline-Discipline\n$\\bar{{r}} = {r_disc:.4f}$", fontweight='bold')
    plt.colorbar(im_b, ax=axes[1], shrink=0.8)

    # (c) Cross-axis Language-Discipline
    cross_corr = np.zeros((n_lang, n_disc))
    for i in range(n_lang):
        for j in range(n_disc):
            cross_corr[i, j] = np.corrcoef(cit_lang[:, i], cit_disc[:, j])[0, 1]
    r_cross = float(np.mean(cross_corr))
    argmin = np.unravel_index(np.argmin(cross_corr), cross_corr.shape)
    min_pair = f'{lang_cats[argmin[0]]}-{disc_cats[argmin[1]]}'
    r_min = float(np.min(cross_corr))

    im_c = axes[2].imshow(cross_corr, cmap=COLORS["correlation"], vmin=0.97, vmax=1.0)
    axes[2].set_xticks(range(n_disc)); axes[2].set_xticklabels(disc_cats, fontsize=10)
    axes[2].set_yticks(range(n_lang)); axes[2].set_yticklabels(lang_cats, fontsize=10)
    axes[2].set_xlabel("Discipline", fontsize=11)
    axes[2].set_ylabel("Language", fontsize=11)
    axes[2].set_title(f"(c) Language-Discipline\n$\\bar{{r}} = {r_cross:.4f}$  (min: {min_pair} $r$={r_min:.4f})",
                      fontweight='bold')
    plt.colorbar(im_c, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


# ── Figure 3: Per-Profile Layer Selection ────────────────────────────

def figure_layer_selection(
    profile_results: Dict[str, Dict],
    output_path: str,
    n_layers: int = 24,
):
    """Generate a figure showing which layers are retained/pruned per profile."""
    apply_style()
    profile_names = sorted(profile_results.keys())
    n_profiles = len(profile_names)

    fig, ax = plt.subplots(figsize=(14, 0.4 * n_profiles + 2))

    matrix = np.zeros((n_profiles, n_layers))
    for i, pname in enumerate(profile_names):
        retained = profile_results[pname].get("retained", [])
        for l in retained:
            l_int = int(l) if not isinstance(l, (int, np.integer)) else int(l)
            matrix[i, l_int] = 1

    cmap = mcolors.ListedColormap(['#f0f0f0', COLORS["lang"]])
    ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_yticks(range(n_profiles))
    ax.set_yticklabels(profile_names, fontsize=9)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(range(n_layers), fontsize=8)
    ax.set_xlabel("Layer Index", fontweight='bold')
    ax.set_ylabel("Profile", fontweight='bold')
    ax.set_title("Layer Selection per Preservation Profile\n(dark = retained, light = pruned)", fontweight='bold')

    # Count annotation
    for i, pname in enumerate(profile_names):
        n_ret = profile_results[pname].get("n_retained", len(profile_results[pname].get("retained", [])))
        ax.text(n_layers + 0.5, i, f'{n_ret}', fontsize=8, va='center', color='gray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


# ── Figure 4: Per-Category CIT Depth Profiles ────────────────────────

def figure_depth_profiles(
    cit_lang: np.ndarray, lang_cats: List[str],
    cit_disc: np.ndarray, disc_cats: List[str],
    cit_scen: np.ndarray, scen_cats: List[str],
    output_path: str,
):
    """Generate per-category CIT score vs layer index line plots."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    for ax, cit, cats, color, title in [
        (axes[0], cit_lang, lang_cats, COLORS["lang"], "Language"),
        (axes[1], cit_disc, disc_cats, COLORS["disc"], "Discipline"),
        (axes[2], cit_scen, scen_cats, COLORS["scen"], "Scenario"),
    ]:
        for i, cat in enumerate(cats):
            ax.plot(range(cit.shape[0]), cit[:, i], '-o', markersize=2,
                    alpha=0.7, label=cat, linewidth=1.2)
        ax.axvspan(0, 5.5, alpha=0.05, color='blue')
        ax.axvspan(15.5, cit.shape[0]-1, alpha=0.05, color='red')
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("CIT Score")
        ax.set_title(f"{title} Axis", fontweight='bold')
        ax.legend(fontsize=6, ncol=2 if len(cats) > 5 else 1, loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return fig


# ── Generate all figures ─────────────────────────────────────────────

def generate_all_figures(
    cit_lang: np.ndarray, lang_cats: List[str],
    cit_disc: np.ndarray, disc_cats: List[str],
    cit_scen: np.ndarray, scen_cats: List[str],
    profile_results: Optional[Dict[str, Dict]] = None,
    output_dir: str = "figures",
    formats: Tuple[str, ...] = ("pdf",),
):
    """Generate all paper figures from real CIT data."""
    os.makedirs(output_dir, exist_ok=True)
    figs = {}

    # Cliff ratios
    cliff = {
        "Language": float((cit_lang[16:].mean(axis=0) / (cit_lang[:6].mean(axis=0) + 1e-8)).mean()),
        "Discipline": float((cit_disc[16:].mean(axis=0) / (cit_disc[:6].mean(axis=0) + 1e-8)).mean()),
        "Scenario": float((cit_scen[16:].mean(axis=0) / (cit_scen[:6].mean(axis=0) + 1e-8)).mean()),
    }

    for fmt in formats:
        figs["fig1"] = figure_cit_heatmap(
            cit_lang, lang_cats, cit_disc, disc_cats, cit_scen, scen_cats,
            f"{output_dir}/fig1_cit_analysis.{fmt}", cliff,
        )
        figs["fig2"] = figure_correlation_matrices(
            cit_lang, lang_cats, cit_disc, disc_cats,
            f"{output_dir}/fig2_correlation.{fmt}",
        )
        figs["fig4"] = figure_depth_profiles(
            cit_lang, lang_cats, cit_disc, disc_cats, cit_scen, scen_cats,
            f"{output_dir}/fig4_depth_profiles.{fmt}",
        )

    if profile_results:
        for fmt in formats:
            figs["fig3"] = figure_layer_selection(
                profile_results, f"{output_dir}/fig3_layer_selection.{fmt}"
            )

    return figs
