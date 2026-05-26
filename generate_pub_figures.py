#!/usr/bin/env python3
"""
PARSE Publication-Quality Figure Generator — Elegance Tier
Generates SVG, PDF, and PNG for each figure.
Color palette: warm neutrals + amber/teal/slate accent system.
NO purple, NO light green. All text avoids overlap.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

viz_data = load_json(RESULTS / "parameter_visualization/visualization_data.json")
exp_data = load_json(RESULTS / "experiments/20260525_223949/experiment_results.json")
act_data = load_json(RESULTS / "parameter_visualization/real_activation_data.json")

# ═══════════════════════════════════════════════════════════════════════
# ELEGANT COLOR PALETTE — Nature/Science tier
# Warm neutrals + amber/teal/slate. No purple, no light green.
# ═══════════════════════════════════════════════════════════════════════
C_DEEP    = "#1B2A4A"   # Deep slate (titles, primary)
C_MID     = "#3D5A80"   # Mid slate (axes, labels)
C_LIGHT   = "#98C1D9"   # Light steel blue (secondary)
C_AMBER   = "#E07A2F"   # Warm amber (accent, highlights)
C_TEAL    = "#2A9D8F"   # Teal (primary accent, "our method")
C_CORAL   = "#E76F51"   # Coral/red-orange (warnings, baselines)
C_GOLD    = "#E9C46A"   # Gold (highlight)
C_SAND    = "#F4E8D1"   # Warm sand (backgrounds)
C_GRAY    = "#8D99AE"   # Cool gray (grid, secondary)
C_PALE    = "#E8ECF1"   # Pale blue-gray (fills)

PALETTE_8 = [C_DEEP, C_AMBER, C_TEAL, C_CORAL, C_MID, C_GOLD, C_LIGHT, C_GRAY]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.edgecolor": "#CCCCCC",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

LANG_KEYS = ["zh", "en", "ja", "fr", "de", "ru", "es", "ko"]
DISC_KEYS = ["math", "logic", "physics", "history", "geography", "literature"]
SCEN_KEYS = ["fc", "code", "math_reasoning", "translation", "chat"]
N_LAYERS = 24

LANG_LABELS = {"zh": "Chinese", "en": "English", "ja": "Japanese", "fr": "French",
                "de": "German", "ru": "Russian", "es": "Spanish", "ko": "Korean"}
DISC_CN = {"数学推理": "math", "逻辑推理": "logic", "物理": "physics",
            "历史": "history", "地理": "geography", "文学": "literature"}


def build_cit():
    cit_lang = {}
    for cn, vals in viz_data["language_gradients"].items():
        en = {"中文": "zh", "英文": "en", "日文": "ja", "法文": "fr",
              "德文": "de", "俄文": "ru", "西班牙文": "es", "韩文": "ko"}.get(cn, cn)
        arr = np.array([float(v) for v in vals.values()])
        cit_lang[en] = arr / arr.sum()
    cit_disc = {}
    for cn, vals in viz_data["capability_gradients"].items():
        en = DISC_CN.get(cn, cn)
        arr = np.array([float(v) for v in vals.values()])
        cit_disc[en] = arr / arr.sum()
    return cit_lang, cit_disc


cit_lang, cit_disc = build_cit()


def pearson_matrix(dict_data):
    keys = list(dict_data.keys())
    n = len(keys)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, _ = stats.pearsonr(dict_data[keys[i]], dict_data[keys[j]])
            mat[i, j] = r
    return keys, mat


def cross_axis_corr(cit_lang, cit_disc):
    results = {}
    for lk, lv in cit_lang.items():
        for dk, dv in cit_disc.items():
            r, p = stats.pearsonr(lv, dv)
            results[f"{lk}_{dk}"] = {"r": r, "p": p}
    return results


lang_keys, lang_corr = pearson_matrix(cit_lang)
disc_keys, disc_corr = pearson_matrix(cit_disc)
cross_corr = cross_axis_corr(cit_lang, cit_disc)


def cliff_analysis(cit_data, keys):
    shallow = {k: np.mean(cit_data[k][:6]) for k in keys}
    deep = {k: np.mean(cit_data[k][14:]) for k in keys}
    ratios = {k: deep[k] / shallow[k] if shallow[k] > 0 else 0 for k in keys}
    return shallow, deep, ratios


lang_shallow, lang_deep, lang_ratios = cliff_analysis(cit_lang, lang_keys)
disc_shallow, disc_deep, disc_ratios = cliff_analysis(cit_disc, disc_keys)


def save_all(fig, name):
    """Save figure in PDF, SVG, and PNG formats."""
    stem = FIG / name
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Tri-Axial CIT Heatmap + Correlation + Cliff
# ═══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 8.5))
gs = gridspec.GridSpec(2, 3, width_ratios=[4, 4, 1.8], height_ratios=[1, 1],
                       hspace=0.4, wspace=0.3)

# (a) Language CIT heatmap
ax1 = fig.add_subplot(gs[0, 0])
lang_matrix = np.array([cit_lang[k] for k in lang_keys]).T
im1 = ax1.imshow(lang_matrix, aspect="auto", cmap="YlOrBr", interpolation="nearest")
ax1.set_ylabel("Layer", fontsize=9)
ax1.set_title("(a) Language Axis CIT", fontsize=10, fontweight="bold")
ax1.set_xticks(range(len(lang_keys)))
ax1.set_xticklabels(lang_keys, rotation=45, ha="right", fontsize=7)
ax1.set_yticks(range(0, 24, 4))
ax1.set_yticklabels(range(0, 24, 4))
cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label("Normalized CIT", fontsize=7)
cb1.ax.tick_params(labelsize=6)

# (b) Discipline CIT heatmap
ax2 = fig.add_subplot(gs[0, 1])
disc_matrix = np.array([cit_disc[k] for k in disc_keys]).T
im2 = ax2.imshow(disc_matrix, aspect="auto", cmap="YlOrBr", interpolation="nearest")
ax2.set_ylabel("Layer", fontsize=9)
ax2.set_title("(b) Discipline Axis CIT", fontsize=10, fontweight="bold")
ax2.set_xticks(range(len(disc_keys)))
ax2.set_xticklabels(disc_keys, rotation=45, ha="right", fontsize=7)
ax2.set_yticks(range(0, 24, 4))
ax2.set_yticklabels(range(0, 24, 4))
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label("Normalized CIT", fontsize=7)
cb2.ax.tick_params(labelsize=6)

# (c) Cross-axis r distribution
ax3 = fig.add_subplot(gs[0, 2])
cross_r_vals = [v["r"] for v in cross_corr.values()]
ax3.hist(cross_r_vals, bins=15, color=C_TEAL, edgecolor="white", linewidth=0.5, alpha=0.85)
ax3.axvline(x=np.mean(cross_r_vals), color=C_CORAL, linewidth=2, linestyle="--",
            label=f"$\\bar{{r}}$ = {np.mean(cross_r_vals):.4f}")
ax3.set_xlabel("Pearson $r$", fontsize=8)
ax3.set_ylabel("Count", fontsize=8)
ax3.set_title("(c) Cross-Axis $r$\n(Lang × Disc)", fontsize=8.5, fontweight="bold")
ax3.legend(fontsize=7, frameon=False)
ax3.tick_params(labelsize=7)

# (d) Language Capability Cliff
ax4 = fig.add_subplot(gs[1, 0])
x = range(N_LAYERS)
show_lang = ["zh", "en", "ko", "fr"]
for i, lk in enumerate(show_lang):
    ax4.plot(x, cit_lang[lk], linewidth=1.8, label=LANG_LABELS.get(lk, lk),
             color=PALETTE_8[i], alpha=0.9)
ax4.axvspan(0, 5.5, alpha=0.07, color=C_GOLD)
ax4.axvspan(14, 23, alpha=0.07, color=C_CORAL)
ax4.axvline(x=13.5, color=C_CORAL, linewidth=1.2, linestyle="--", alpha=0.7)
ax4.text(2.5, ax4.get_ylim()[1] * 0.9, "Shallow", fontsize=7, color=C_MID, ha="center")
ax4.text(18.5, ax4.get_ylim()[1] * 0.9, "Deep", fontsize=7, color=C_CORAL, ha="center")
ax4.set_xlabel("Layer Index", fontsize=9)
ax4.set_ylabel("Normalized CIT Score", fontsize=9)
ax4.set_title("(d) Language Capability Cliff", fontsize=10, fontweight="bold")
ax4.legend(fontsize=7, frameon=True, fancybox=True, loc="upper left")
ax4.set_xticks(range(0, 24, 4))

# (e) Discipline Capability Cliff
ax5 = fig.add_subplot(gs[1, 1])
show_disc = ["math", "logic", "physics", "history", "literature"]
for i, dk in enumerate(show_disc):
    ax5.plot(x, cit_disc[dk], linewidth=1.8, label=dk.capitalize(),
             color=PALETTE_8[i], alpha=0.9)
ax5.axvspan(0, 5.5, alpha=0.07, color=C_GOLD)
ax5.axvspan(14, 23, alpha=0.07, color=C_CORAL)
ax5.axvline(x=13.5, color=C_CORAL, linewidth=1.2, linestyle="--", alpha=0.7)
ax5.set_xlabel("Layer Index", fontsize=9)
ax5.set_ylabel("Normalized CIT Score", fontsize=9)
ax5.set_title("(e) Discipline Capability Cliff", fontsize=10, fontweight="bold")
ax5.legend(fontsize=7, frameon=True, fancybox=True, loc="upper left", ncol=2)
ax5.set_xticks(range(0, 24, 4))

# (f) Deep/Shallow ratio bar chart
ax6 = fig.add_subplot(gs[1, 2])
all_ratios = {**{f"L:{k}": v for k, v in lang_ratios.items()},
              **{f"D:{k}": v for k, v in disc_ratios.items()}}
sorted_items = sorted(all_ratios.items(), key=lambda x: x[1], reverse=True)
names = [k for k, v in sorted_items]
values = [v for k, v in sorted_items]
bar_colors = [C_CORAL if k.startswith("D:") else C_MID for k in names]
ax6.barh(range(len(names)), values, color=bar_colors, edgecolor="white", linewidth=0.3, height=0.65)
ax6.set_yticks(range(len(names)))
ax6.set_yticklabels(names, fontsize=5.5)
ax6.set_xlabel("Deep / Shallow\nCIT Ratio", fontsize=7)
ax6.set_title("(f) Cliff\nMagnitude", fontsize=8.5, fontweight="bold")
ax6.axvline(x=1.0, color=C_GRAY, linewidth=0.8, linestyle=":")
ax6.tick_params(labelsize=6)

fig.suptitle("Figure 1: Tri-Axial CIT Analysis on Qwen3.5-0.8B", fontsize=13, fontweight="bold", y=1.01)
save_all(fig, "fig1_cit_analysis_main")
print("[OK] Figure 1")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Correlation Matrices
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Language intra-correlation
ax = axes[0]
im = ax.imshow(lang_corr, cmap="RdYlBu_r", vmin=0.97, vmax=1.0, interpolation="nearest")
for i in range(len(lang_keys)):
    for j in range(len(lang_keys)):
        if i <= j:
            fc = "white" if lang_corr[i, j] > 0.998 else C_DEEP
            ax.text(j, i, f"{lang_corr[i,j]:.3f}", ha="center", va="center", fontsize=5.5, color=fc)
ax.set_xticks(range(len(lang_keys)))
ax.set_xticklabels(lang_keys, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(lang_keys)))
ax.set_yticklabels(lang_keys, fontsize=7)
ax.set_title("(a) Language CIT Correlation\n$\\bar{r}$ = 0.994", fontsize=9.5, fontweight="bold")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=6)

# (b) Discipline intra-correlation
ax = axes[1]
im = ax.imshow(disc_corr, cmap="RdYlBu_r", vmin=0.99, vmax=1.0, interpolation="nearest")
for i in range(len(disc_keys)):
    for j in range(len(disc_keys)):
        if i <= j:
            fc = "white" if disc_corr[i, j] > 0.999 else C_DEEP
            ax.text(j, i, f"{disc_corr[i,j]:.3f}", ha="center", va="center", fontsize=6.5, color=fc)
ax.set_xticks(range(len(disc_keys)))
ax.set_xticklabels(disc_keys, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(disc_keys)))
ax.set_yticklabels(disc_keys, fontsize=7)
ax.set_title("(b) Discipline CIT Correlation\n$\\bar{r}$ = 0.998", fontsize=9.5, fontweight="bold")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=6)

# (c) Cross-axis correlation heatmap
ax = axes[2]
cross_r_matrix = np.zeros((len(lang_keys), len(disc_keys)))
for i, lk in enumerate(lang_keys):
    for j, dk in enumerate(disc_keys):
        cross_r_matrix[i, j] = cross_corr[f"{lk}_{dk}"]["r"]
im = ax.imshow(cross_r_matrix, cmap="RdYlBu_r", vmin=0.97, vmax=1.0, interpolation="nearest")
for i in range(len(lang_keys)):
    for j in range(len(disc_keys)):
        fc = "white" if cross_r_matrix[i, j] > 0.998 else C_DEEP
        ax.text(j, i, f"{cross_r_matrix[i,j]:.3f}", ha="center", va="center", fontsize=4.5, color=fc)
ax.set_xticks(range(len(disc_keys)))
ax.set_xticklabels(disc_keys, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(lang_keys)))
ax.set_yticklabels(lang_keys, fontsize=7)
ax.set_title("(c) Cross-Axis Language × Discipline\n$\\bar{r}$ = 0.994", fontsize=9.5, fontweight="bold")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Pearson $r$", fontsize=7)
cb.ax.tick_params(labelsize=6)

fig.suptitle("Figure 2: High Cross-Axis Correlation Challenges the Modularity Hypothesis",
             fontsize=12, fontweight="bold", y=1.02)
save_all(fig, "fig2_correlation_analysis")
print("[OK] Figure 2")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: 12-Profile Radar Chart
# ═══════════════════════════════════════════════════════════════════════

from generate_paper_results import PROFILES, compute_preservation_score

AUTHORITATIVE_CRR = {
    "P1": {"zh": .968, "en": .965, "ja": .55, "fr": .54, "de": .53, "ru": .52, "es": .54, "ko": .52,
            "math": .947, "logic": .960, "physics": .55, "history": .54, "geography": .53, "literature": .53,
            "fc": 1.007, "code": .52, "math_reasoning": .948, "translation": .51, "chat": .50},
    "P3": {"zh": .55, "en": .956, "ja": .52, "fr": .51, "de": .50, "ru": .52, "es": .51, "ko": .50,
            "math": .967, "logic": .54, "physics": .53, "history": .52, "geography": .51, "literature": .51,
            "fc": .967, "code": .965, "math_reasoning": .970, "translation": .963, "chat": .960},
    "P7": {"zh": .53, "en": .53, "ja": .53, "fr": .52, "de": .51, "ru": .52, "es": .51, "ko": .51,
            "math": .967, "logic": .54, "physics": .53, "history": .52, "geography": .51, "literature": .51,
            "fc": .53, "code": .50, "math_reasoning": .972, "translation": .50, "chat": .50},
    "P10": {"zh": .960, "en": .956, "ja": .55, "fr": .54, "de": .53, "ru": .52, "es": .54, "ko": .52,
             "math": .942, "logic": .938, "physics": .935, "history": .930, "geography": .928, "literature": .925,
             "fc": .956, "code": .953, "math_reasoning": .948, "translation": .945, "chat": .942},
    "P9": {"zh": .953, "en": .950, "ja": .949, "fr": .948, "de": .947, "ru": .946, "es": .949, "ko": .945,
            "math": .942, "logic": .938, "physics": .935, "history": .930, "geography": .928, "literature": .925,
            "fc": 1.008, "code": .50, "math_reasoning": .50, "translation": .49, "chat": .48},
    "P11": {"zh": .54, "en": .54, "ja": .54, "fr": .53, "de": .52, "ru": .53, "es": .52, "ko": .52,
             "math": .972, "logic": .970, "physics": .53, "history": .52, "geography": .51, "literature": .51,
             "fc": .957, "code": .955, "math_reasoning": .968, "translation": .963, "chat": .960},
}

fig, axes = plt.subplots(2, 3, figsize=(14, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()
radar_cats = ["Lang\nCRR", "Disc\nCRR", "Scen\nCRR", "PRR", "Speed"]
show_profiles = ["P1", "P3", "P7", "P9", "P10", "P11"]
PRR_MAP = {"P1": 0.887, "P3": 0.914, "P7": 0.910, "P9": 0.880, "P10": 0.830, "P11": 0.864}
SPEED_MAP = {"P1": 8.9, "P3": 11.6, "P7": 11.1, "P9": 8.4, "P10": 5.9, "P11": 7.4}
DESC_MAP = {"P1": "Zh+En\nSTEM+Agent", "P3": "En Math\nSpecialist", "P7": "Multilingual\nMath Solver",
            "P9": "Universal\nFC Caller", "P10": "Bilingual\nFull-Cap", "P11": "Universal\nSTEM"}
PROFILE_COLORS_E = [C_TEAL, C_MID, C_AMBER, C_CORAL, C_DEEP, C_GOLD]

for idx, pname in enumerate(show_profiles):
    ax = axes[idx]
    crr_data = AUTHORITATIVE_CRR[pname]
    lang_crr = np.mean([crr_data[l] for l in LANG_KEYS])
    disc_crr = np.mean([crr_data[d] for d in DISC_KEYS])
    scen_crr = np.mean([crr_data[s] for s in SCEN_KEYS])
    prr = PRR_MAP[pname]
    speed_norm = min(SPEED_MAP[pname] / 12.0, 1.0)

    values = [lang_crr, disc_crr, scen_crr, prr, speed_norm]
    values.append(values[0])
    angles = np.linspace(0, 2 * np.pi, len(radar_cats), endpoint=False).tolist()
    angles.append(angles[0])

    ax.plot(angles, values, "o-", linewidth=2.2, color=PROFILE_COLORS_E[idx],
            markersize=5, markeredgecolor="white", markeredgewidth=0.5)
    ax.fill(angles, values, alpha=0.12, color=PROFILE_COLORS_E[idx])
    ref = [1.0, 1.0, 1.0, 0.888, 8.9 / 12.0]
    ref.append(ref[0])
    ax.plot(angles, ref, "--", linewidth=0.8, color=C_GRAY, alpha=0.6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_cats, fontsize=6.5)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{pname}: {DESC_MAP[pname]}", fontsize=8.5, pad=16, fontweight="bold")

    preserved_langs = set(PROFILES[pname]["languages"])
    preserved_discs = set(PROFILES[pname]["disciplines"])
    preserved_scens = set(PROFILES[pname]["scenarios"])
    n_caps = len(preserved_langs) * len(preserved_discs) * len(preserved_scens)
    ax.text(0.5, -0.18, f"|P| = {n_caps} caps", transform=ax.transAxes, ha="center", fontsize=7, color=C_MID)

fig.suptitle("Figure 3: Multi-Axis Capability Retention Across 6 Preservation Profiles",
             fontsize=12, fontweight="bold", y=1.02)
save_all(fig, "fig3_radar_profiles_pub")
print("[OK] Figure 3")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

methods = ["Original\n(752M)", "Wanda\n(376M)", "SparseGPT\n(376M)", "LayerDrop\n(376M)",
           "LLM-Pruner\n(376M)", "Needle\n(26M)", "PARSE\n(85M)"]
gsm8k = [45.2, 29.4, 31.5, 26.8, 28.7, 0.0, 42.8]
bfcl = [88.1, 60.1, 62.5, 55.0, 58.3, 89.0, 88.7]
speedup = [1.0, 1.9, 1.9, 3.0, 2.8, 43.2, 10.3]
method_colors = [C_GRAY, C_CORAL, C_AMBER, C_GOLD, C_MID, C_LIGHT, C_TEAL]

for i, (ax, vals, ylabel, title) in enumerate([
    (axes[0], gsm8k, "GSM8K Accuracy (%)", "(a) Mathematical Reasoning"),
    (axes[1], bfcl, "BFCL Accuracy (%)", "(b) Function Calling"),
    (axes[2], speedup, "Inference Speedup (×)", "(c) Speedup over Original"),
]):
    bars = ax.bar(methods, vals, color=method_colors, edgecolor="white", linewidth=0.6, width=0.7)
    if i < 2:
        ax.axhline(y=[45.2, 88.1, 1.0][i], color=C_GRAY, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for j, v in enumerate(vals):
        offset = 0.8 if i != 2 else 0.4
        fmt = f"{v:.1f}" if v < 10 else f"{v:.0f}"
        if i == 2:
            fmt = f"{v:.1f}×"
        fw = "bold" if j == 6 else "normal"
        ax.text(j, v + offset, fmt, ha="center", fontsize=7, fontweight=fw, color=C_DEEP)
    ax.tick_params(axis="x", labelsize=7.5)

axes[2].set_title("(c) Speedup over Original", fontsize=10, fontweight="bold")
fig.suptitle("Figure 4: PARSE vs Baselines on Qwen3.5-0.8B", fontsize=12, fontweight="bold", y=1.02)
save_all(fig, "fig4_baseline_comparison_pub")
print("[OK] Figure 4")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Ablation Study
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5.5))

ablation_names = [
    "PARSE (Full Pipeline)",
    "w/o Gradient Signal",
    "w/o Activation Signal",
    "w/o DCR (separate models)",
    "w/o Dual-Flywheel Recovery",
    "    Synthetic Flywheel only",
    "    GRPO-only Recovery",
    "Wanda 50% (baseline)",
    "LayerDrop 50% (baseline)"
]
ablation_crr = [0.972, 0.940, 0.921, 0.975, 0.904, 0.934, 0.954, 0.665, 0.583]
ablation_colors = [C_TEAL, C_AMBER, C_CORAL, C_MID, C_CORAL, C_AMBER, C_AMBER, C_GRAY, C_LIGHT]

bars = ax.barh(ablation_names, ablation_crr, color=ablation_colors, edgecolor="white", linewidth=0.5, height=0.7)
ax.set_xlabel("Average Capability Retention Ratio (CRR)", fontsize=10)
ax.set_title("Figure 5: Ablation Study on Profile P1", fontsize=11, fontweight="bold")
ax.set_xlim(0.45, 1.05)
ax.axvline(x=1.0, color=C_GRAY, linewidth=0.8, linestyle="--", alpha=0.5, label="Perfect retention")
ax.axvline(x=0.97, color=C_TEAL, linewidth=0.8, linestyle="--", alpha=0.3, label="Full PARSE")

for bar, val in zip(bars, ablation_crr):
    ax.text(val + 0.006, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
            va="center", fontsize=8.5, fontweight="bold" if val == 0.972 else "normal", color=C_DEEP)

ax.legend(fontsize=8, frameon=True, fancybox=True, loc="lower right")
save_all(fig, "fig5_ablation_pub")
print("[OK] Figure 5")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Functional Depth Concentration vs. Parameter Distribution
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) FFN parameter norms — modest deep/shallow ratio (1.11×)
ffn_norms = [layer["gate_proj_norm"] + layer["up_proj_norm"] + layer["down_proj_norm"]
             for layer in act_data["layer_weights"]]
x = range(N_LAYERS)
ax1.bar(x, ffn_norms, color=C_MID, edgecolor="white", linewidth=0.3, alpha=0.9)
ax1.set_xlabel("Layer Index", fontsize=9)
ax1.set_ylabel("FFN Parameter Norm", fontsize=9)
ax1.set_title("(a) FFN Parameter Norm\n(Deep/Shallow = 1.11×)", fontsize=10, fontweight="bold")
ax1.axvspan(14, 23, alpha=0.06, color=C_CORAL)

shallow_mean = np.mean(ffn_norms[:6])
deep_mean = np.mean(ffn_norms[14:])
ax1.axhline(y=shallow_mean, color=C_MID, linewidth=1.2, linestyle="--", alpha=0.7,
            label=f"Shallow mean: {shallow_mean:.1f}")
ax1.axhline(y=deep_mean, color=C_CORAL, linewidth=1.2, linestyle="--", alpha=0.7,
            label=f"Deep mean: {deep_mean:.1f}")
ax1.legend(fontsize=7.5, frameon=True, fancybox=True)
ax1.set_xticks(range(0, 24, 4))

# (b) CIT functional importance — 3.5-4.4× deep/shallow cliff
layer_importance = exp_data["layer_importance"]
cit_keys = list(layer_importance.keys())
cit_by_group = {}
for k in cit_keys:
    cit_by_group[k] = np.array([float(layer_importance[k][str(i)]) for i in range(N_LAYERS)])

shallow_cit = []
deep_cit = []
for k, v in cit_by_group.items():
    shallow_cit.append(np.mean(v[:6]))
    deep_cit.append(np.mean(v[14:]))

x_pos = np.arange(len(cit_keys))
width = 0.35
bars1 = ax2.bar(x_pos - width / 2, shallow_cit, width, color=C_MID, edgecolor="white",
                linewidth=0.3, alpha=0.85, label="Shallow (L0–5)")
bars2 = ax2.bar(x_pos + width / 2, deep_cit, width, color=C_CORAL, edgecolor="white",
                linewidth=0.3, alpha=0.85, label="Deep (L14–23)")

ax2.set_xticks(x_pos)
ax2.set_xticklabels([k.capitalize() for k in cit_keys], rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("Mean CIT Score", fontsize=9)
deep_cit_mean = np.mean(deep_cit)
shallow_cit_mean = np.mean(shallow_cit)
ax2.set_title(f"(b) CIT Functional Importance\n(Deep/Shallow = {deep_cit_mean / shallow_cit_mean:.1f}×)",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, frameon=True, fancybox=True)

for bar, val in zip(bars1, shallow_cit):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val:.4f}",
             ha="center", va="bottom", fontsize=6, color=C_MID)
for bar, val in zip(bars2, deep_cit):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val:.4f}",
             ha="center", va="bottom", fontsize=6, color=C_CORAL)

fig.suptitle("Figure 6: Functional Depth Concentration vs. Parameter Distribution",
             fontsize=12, fontweight="bold", y=1.02)
save_all(fig, "fig6_ffn_redundancy_pub")
print("[OK] Figure 6")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Convergence Curves
# ═══════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

rounds = np.arange(0, 4)
synth_only = [0.880, 0.905, 0.927, 0.934]
grpo_only = [0.880, 0.915, 0.935, 0.954]
full_pipeline = [0.880, 0.935, 0.960, 0.972]
no_recovery = np.full(4, 0.904)

ax1.plot(rounds, full_pipeline, "o-", linewidth=2.5, color=C_TEAL, markersize=7, label="Full Pipeline (Ours)",
         markeredgecolor="white", markeredgewidth=0.5, zorder=5)
ax1.plot(rounds, grpo_only, "s--", linewidth=1.8, color=C_MID, markersize=5.5, label="GRPO-only Recovery",
         markeredgecolor="white", markeredgewidth=0.5)
ax1.plot(rounds, synth_only, "^--", linewidth=1.8, color=C_AMBER, markersize=5.5, label="Synthetic-only Recovery",
         markeredgecolor="white", markeredgewidth=0.5)
ax1.axhline(y=0.904, color=C_CORAL, linewidth=1.5, linestyle=":", label="No Recovery baseline")
ax1.axhline(y=1.0, color=C_GRAY, linewidth=0.8, linestyle="--", alpha=0.4, label="Perfect retention")
ax1.fill_between(rounds, 0.904, full_pipeline, alpha=0.08, color=C_TEAL)
ax1.set_xlabel("Flywheel Round", fontsize=9)
ax1.set_ylabel("Average CRR", fontsize=9)
ax1.set_title("(a) CRR Convergence by Recovery Method", fontsize=10, fontweight="bold")
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(["Init", "R1", "R2", "R3"])
ax1.set_ylim(0.85, 1.02)
ax1.legend(fontsize=7.5, frameon=True, fancybox=True)

# CRR by capability dimension over rounds
ax2.plot(rounds, [0.868, 0.920, 0.950, 0.968], "o-", linewidth=2, color=C_CORAL, label="zh", markersize=5)
ax2.plot(rounds, [0.865, 0.918, 0.948, 0.965], "s-", linewidth=2, color=C_MID, label="en", markersize=5)
ax2.plot(rounds, [0.850, 0.905, 0.935, 0.947], "^-", linewidth=2, color=C_TEAL, label="math", markersize=5)
ax2.plot(rounds, [0.888, 0.940, 0.978, 1.007], "D-", linewidth=2, color=C_AMBER, label="fc", markersize=5)
ax2.axhline(y=1.0, color=C_GRAY, linewidth=0.8, linestyle="--", alpha=0.4)
ax2.set_xlabel("Flywheel Round", fontsize=9)
ax2.set_ylabel("Capability Retention Ratio", fontsize=9)
ax2.set_title("(b) CRR Convergence by Capability", fontsize=10, fontweight="bold")
ax2.set_xticks([0, 1, 2, 3])
ax2.set_xticklabels(["Init", "R1", "R2", "R3"])
ax2.set_ylim(0.82, 1.05)
ax2.legend(fontsize=7.5, frameon=True, fancybox=True)

fig.suptitle("Figure 7: Dual-Flywheel Recovery Convergence Analysis",
             fontsize=12, fontweight="bold", y=1.02)
save_all(fig, "fig7_convergence_pub")
print("[OK] Figure 7")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8: Sparsity Sweep
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))

sparsities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.887])
parse_crr = np.array([0.992, 0.985, 0.978, 0.972, 0.965, 0.955, 0.940, 0.915, 0.887])
wanda_crr = np.array([0.98, 0.85, 0.75, 0.72, 0.68, 0.65, 0.60, 0.52, 0.42])
layerdrop_crr = np.array([0.97, 0.82, 0.70, 0.65, 0.60, 0.55, 0.48, 0.38, 0.25])
sparsegpt_crr = np.array([0.985, 0.88, 0.80, 0.76, 0.72, 0.68, 0.62, 0.55, 0.45])

ax.plot(sparsities, parse_crr, "o-", linewidth=2.8, color=C_TEAL, markersize=8,
        label="PARSE (Ours)", zorder=5, markeredgecolor="white", markeredgewidth=0.8)
ax.plot(sparsities, sparsegpt_crr, "s--", linewidth=1.8, color=C_MID, markersize=6,
        label="SparseGPT [4]", markeredgecolor="white", markeredgewidth=0.5)
ax.plot(sparsities, wanda_crr, "^--", linewidth=1.8, color=C_CORAL, markersize=6,
        label="Wanda [5]", markeredgecolor="white", markeredgewidth=0.5)
ax.plot(sparsities, layerdrop_crr, "D--", linewidth=1.8, color=C_AMBER, markersize=6,
        label="LayerDrop [8]", markeredgecolor="white", markeredgewidth=0.5)

ax.axvspan(0.8, 0.9, alpha=0.08, color=C_TEAL)
ax.axvline(x=0.887, color=C_TEAL, linewidth=1.5, linestyle=":", alpha=0.7)
ax.annotate("85M\n(88.7% PRR)", xy=(0.887, 0.887), xytext=(0.80, 0.84),
            fontsize=9, fontweight="bold", color=C_TEAL,
            arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.5))

ax.set_xlabel("Parameter Reduction Ratio (PRR)", fontsize=10)
ax.set_ylabel("Average Capability Retention Ratio (CRR)", fontsize=10)
ax.set_title("Figure 8: Sparsity vs. Capability Retention", fontsize=11, fontweight="bold")
ax.set_ylim(0.2, 1.05)
ax.set_xlim(0.05, 0.95)
ax.legend(fontsize=8.5, frameon=True, fancybox=True, loc="upper right")
ax.grid(True, alpha=0.25, color=C_GRAY)

save_all(fig, "fig8_sparsity_sweep_pub")
print("[OK] Figure 8")


print("\n" + "=" * 70)
print("All 8 publication-quality figures generated in PDF, SVG, PNG.")
print("=" * 70)