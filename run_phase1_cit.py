"""
Phase 1: CIT Computation Experiment
Runs Stage 1 (Diagnostic Probing) on Qwen3.5-0.8B using local Apple M4.
Generates: CIT matrices, correlation data, layer selection results.
Memory target: < 8GB unified memory (well within 16GB limit).
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from parse.config import PROFILES, LANGUAGES, DISCIPLINES, SCENARIOS
from parse.data.calibration import build_default_calibration
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config ----
MODEL_PATH = "models/qwen/Qwen3___5-0___8B"
DEVICE = "mps"  # Apple Silicon GPU
TORCH_DTYPE = torch.float16
ALPHA = 0.6
OUTPUT_DIR = "results/experiments/phase1_cit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 1: CIT Computation Experiment")
print(f"Device: {DEVICE} | Dtype: {TORCH_DTYPE}")
print(f"Model: {MODEL_PATH}")
print("=" * 60)

# ---- Load Model ----
print("\n[1/5] Loading model...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE,
    local_files_only=True,
)
model.eval()
print(f"  Model loaded in {time.time()-t0:.1f}s")

# Detect layers (Qwen3.5 uses model.language_model.layers)
for attr in ["model.language_model.layers", "model.layers", "language_model.layers"]:
    parts = attr.split(".")
    obj = model
    try:
        for p in parts:
            obj = getattr(obj, p)
        layers = list(obj)
        N_LAYERS = len(layers)
        break
    except (AttributeError, TypeError):
        continue
else:
    raise RuntimeError("Cannot find transformer layers")

# Hidden size from first layer
HIDDEN_SIZE = None
for n, p in layers[0].named_parameters():
    if len(p.shape) == 2 and "weight" in n:
        HIDDEN_SIZE = p.shape[1]
        break
if HIDDEN_SIZE is None:
    HIDDEN_SIZE = 1024
print(f"  Layers: {N_LAYERS} | Hidden: {HIDDEN_SIZE}")

# ---- Load Calibration Data ----
print("\n[2/5] Loading calibration data...")
cal = build_default_calibration()
print(f"  Lang: {len(cal.lang)} categories | Disc: {len(cal.disc)} | Scen: {len(cal.scen)}")

# ---- CIT Computation ----
# Import after model load to keep imports clean
from collections import defaultdict

print("\n[3/5] Computing CIT marginals...")
t0 = time.time()

def get_transformer_layers(model):
    """Get transformer layers, handling Qwen3.5's language_model wrapper."""
    for attr in ["model.language_model.layers", "model.layers", "language_model.layers"]:
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            return list(obj)
        except (AttributeError, TypeError):
            continue
    return list(model.model.layers)

def compute_marginal(calibration_data, axis_name):
    """Compute marginal CIT with careful memory management."""
    categories = list(calibration_data.keys())
    n_categories = len(categories)
    cit_marginal = torch.zeros(N_LAYERS, n_categories)
    ffn_keywords = ["gate_proj", "up_proj", "down_proj", "mlp"]
    
    for c_idx, (cat_name, prompts) in enumerate(calibration_data.items()):
        if not prompts:
            continue
        print(f"    [{axis_name}] {cat_name} ({len(prompts)} prompts)...", end=" ", flush=True)
        
        # Encode prompts in small batches if needed
        encodings = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(DEVICE)
        
        # --- Activation capacitance ---
        hooks = []
        layer_acts = defaultdict(float)
        
        def hook_fn(layer_idx):
            def fn(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                layer_acts[layer_idx] += out.detach().abs().sum().item()
            return fn
        
        xf_layers = get_transformer_layers(model)
        for i, layer in enumerate(xf_layers):
            h = layer.register_forward_hook(hook_fn(i))
            hooks.append(h)
        
        with torch.no_grad():
            _ = model(**encodings)
        
        for h in hooks:
            h.remove()
        
        # --- Gradient sensitivity ---
        params = []
        param_to_layer = {}
        for i, layer in enumerate(xf_layers):
            for name, p in layer.named_parameters():
                if p.requires_grad and "weight" in name:
                    if any(kw in name.lower() for kw in ffn_keywords):
                        params.append(p)
                        param_to_layer[id(p)] = i
        
        labels = encodings["input_ids"].clone()
        outputs = model(**encodings, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        layer_grads = defaultdict(float)
        for p in params:
            if p.grad is not None:
                g = (p.grad * p.data).abs().sum().item()
                layer_grads[param_to_layer[id(p)]] += g
        
        model.zero_grad()
        
        # --- Combine ---
        for l in range(N_LAYERS):
            cit_marginal[l, c_idx] = ALPHA * layer_acts[l] + (1 - ALPHA) * layer_grads[l]
        
        # Clean up
        del encodings, outputs, loss
        if DEVICE == "mps":
            torch.mps.empty_cache()
        
        print(f"done")
    
    # Normalize per-category across layers
    cit_marginal = cit_marginal / (cit_marginal.sum(dim=0, keepdim=True) + 1e-8)
    return cit_marginal, categories

cit_lang, lang_cats = compute_marginal(cal.lang, "lang")
cit_disc, disc_cats = compute_marginal(cal.disc, "disc")
cit_scen, scen_cats = compute_marginal(cal.scen, "scen")

print(f"\n  CIT computation completed in {time.time()-t0:.1f}s")

# ---- Correlation Analysis ----
print("\n[4/5] Computing correlation matrices...")

def compute_corr_matrix(cit_matrix, cat_names):
    """Compute pairwise Pearson correlation matrix."""
    n = cit_matrix.shape[1]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(cit_matrix[:, i], cit_matrix[:, j])[0, 1]
    return corr

def mean_pairwise_r(cit_matrix):
    n = cit_matrix.shape[1]
    rs = []
    for i in range(n):
        for j in range(i + 1, n):
            rs.append(np.corrcoef(cit_matrix[:, i], cit_matrix[:, j])[0, 1])
    return np.mean(rs)

def cross_axis_r(cit_a, cit_b):
    rs = []
    for i in range(cit_a.shape[1]):
        for j in range(cit_b.shape[1]):
            rs.append(np.corrcoef(cit_a[:, i], cit_b[:, j])[0, 1])
    return np.mean(rs), rs

# Within-axis correlations
r_lang = mean_pairwise_r(cit_lang)
r_disc = mean_pairwise_r(cit_disc)
r_scen = mean_pairwise_r(cit_scen)

# Cross-axis correlations
r_lang_disc, _ = cross_axis_r(cit_lang, cit_disc)
r_lang_scen, _ = cross_axis_r(cit_lang, cit_scen)
r_disc_scen, _ = cross_axis_r(cit_disc, cit_scen)
r_cross_mean = (r_lang_disc + r_lang_scen + r_disc_scen) / 3

print(f"  Within-axis Pearson r:")
print(f"    Language-Language:  r̄ = {r_lang:.4f}")
print(f"    Discipline-Discipline: r̄ = {r_disc:.4f}")
print(f"    Scenario-Scenario:  r̄ = {r_scen:.4f}")
print(f"  Cross-axis Pearson r:")
print(f"    Language-Discipline: r̄ = {r_lang_disc:.4f}")
print(f"    Language-Scenario:   r̄ = {r_lang_scen:.4f}")
print(f"    Discipline-Scenario: r̄ = {r_disc_scen:.4f}")
print(f"  Mean cross-axis: r̄ = {r_cross_mean:.4f}")

# ---- Layer Selection for all 12 Profiles ----
print("\n[5/5] Computing layer selection for all 12 profiles...")
from parse.config import get_profile

def combine_to_full(cit_lang, cit_disc, cit_scen, profile_langs, profile_discs, profile_scens,
                    lang_cats, disc_cats, scen_cats):
    """Factorized combination of marginal CIT scores."""
    lang_idx = [lang_cats.index(l) for l in profile_langs if l in lang_cats]
    disc_idx = [disc_cats.index(d) for d in profile_discs if d in disc_cats]
    scen_idx = [scen_cats.index(s) for s in profile_scens if s in scen_cats]
    S = torch.zeros(N_LAYERS)
    for li in lang_idx:
        for di in disc_idx:
            for si in scen_idx:
                S += cit_lang[:, li] * cit_disc[:, di] * cit_scen[:, si]
    return S

def select_layers(S_preserve, target_sparsity, standard_attn_layers=None):
    """Select layers to retain/prune."""
    if standard_attn_layers is None:
        standard_attn_layers = {3, 7, 11, 15, 19, 23}
    K = max(1, int(N_LAYERS * (1 - target_sparsity / 2)))
    forced_retain = standard_attn_layers & set(range(N_LAYERS))
    sorted_idx = torch.argsort(S_preserve, descending=True)
    retained = set(forced_retain)
    for idx in sorted_idx.tolist():
        if len(retained) >= K:
            break
        retained.add(idx)
    pruned = [i for i in range(N_LAYERS) if i not in retained]
    return sorted(retained), sorted(pruned)

# Compute deep/shallow CIT ratios (Capability Cliff quantification)
def deep_shallow_ratio(cit_matrix):
    """Ratio of mean CIT in deep layers (16-23) vs shallow (0-5)."""
    deep = cit_matrix[16:24, :].mean(dim=0)
    shallow = cit_matrix[0:6, :].mean(dim=0)
    ratios = deep / (shallow + 1e-8)
    return ratios.numpy()

lang_cliff = deep_shallow_ratio(cit_lang)
disc_cliff = deep_shallow_ratio(cit_disc)
scen_cliff = deep_shallow_ratio(cit_scen)

print(f"\n  Capability Cliff (deep/shallow CIT ratio):")
print(f"    Language:    {lang_cliff.mean():.2f}x (range {lang_cliff.min():.2f}-{lang_cliff.max():.2f})")
print(f"    Discipline:  {disc_cliff.mean():.2f}x (range {disc_cliff.min():.2f}-{disc_cliff.max():.2f})")
print(f"    Scenario:    {scen_cliff.mean():.2f}x (range {scen_cliff.min():.2f}-{scen_cliff.max():.2f})")

profile_results = {}
for pname in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]:
    prof = get_profile(pname)
    S = combine_to_full(cit_lang, cit_disc, cit_scen,
                        prof.languages, prof.disciplines, prof.scenarios,
                        lang_cats, disc_cats, scen_cats)
    retained, pruned = select_layers(S, 0.5)
    xf_layers = get_transformer_layers(model)
    n_ffn_params = sum(
        sum(p.numel() for n, p in xf_layers[i].named_parameters() 
            if any(k in n.lower() for k in ["gate_proj", "up_proj", "down_proj", "mlp"]))
        for i in pruned
    )
    n_total = sum(p.numel() for p in model.parameters())
    compressed = n_total - n_ffn_params
    prr = n_ffn_params / n_total
    
    profile_results[pname] = {
        "description": prof.description,
        "languages": prof.languages,
        "disciplines": prof.disciplines,
        "scenarios": prof.scenarios,
        "retained": retained,
        "pruned": pruned,
        "n_retained": len(retained),
        "n_pruned": len(pruned),
        "ffn_params_removed": n_ffn_params,
        "total_params": n_total,
        "compressed_params": compressed,
        "PRR": round(prr, 4),
        "S_preserve": S.tolist(),
    }
    print(f"  {pname}: {len(retained)} retained, {len(pruned)} pruned, PRR={prr:.1%} | {prof.description}")

# ---- Save Results ----
print(f"\n[Save] Writing results to {OUTPUT_DIR}/")
results = {
    "metadata": {
        "model": "Qwen3.5-0.8B",
        "device": DEVICE,
        "dtype": str(TORCH_DTYPE),
        "n_layers": N_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "cit_alpha": ALPHA,
        "standard_attn_layers": [3, 7, 11, 15, 19, 23],
    },
    "calibration": {
        "lang_categories": lang_cats,
        "disc_categories": disc_cats,
        "scen_categories": scen_cats,
    },
    "cit_matrices": {
        "lang": {cat: cit_lang[:, i].tolist() for i, cat in enumerate(lang_cats)},
        "disc": {cat: cit_disc[:, i].tolist() for i, cat in enumerate(disc_cats)},
        "scen": {cat: cit_scen[:, i].tolist() for i, cat in enumerate(scen_cats)},
    },
    "correlations": {
        "within_axis": {
            "lang_lang": round(r_lang, 4),
            "disc_disc": round(r_disc, 4),
            "scen_scen": round(r_scen, 4),
        },
        "cross_axis": {
            "lang_disc": round(r_lang_disc, 4),
            "lang_scen": round(r_lang_scen, 4),
            "disc_scen": round(r_disc_scen, 4),
            "mean": round(r_cross_mean, 4),
        },
    },
    "capability_cliff": {
        "lang": {cat: round(v, 4) for cat, v in zip(lang_cats, lang_cliff)},
        "disc": {cat: round(v, 4) for cat, v in zip(disc_cats, disc_cliff)},
        "scen": {cat: round(v, 4) for cat, v in zip(scen_cats, scen_cliff)},
        "lang_mean": round(lang_cliff.mean(), 2),
        "disc_mean": round(disc_cliff.mean(), 2),
        "scen_mean": round(scen_cliff.mean(), 2),
    },
    "profiles": profile_results,
}

with open(os.path.join(OUTPUT_DIR, "cit_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save CSV files for paper tables
import csv

# CIT language CSV
with open(os.path.join(OUTPUT_DIR, "cit_language.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["layer"] + lang_cats)
    for l in range(N_LAYERS):
        w.writerow([l] + [f"{cit_lang[l, i]:.6f}" for i in range(len(lang_cats))])

# CIT discipline CSV
with open(os.path.join(OUTPUT_DIR, "cit_discipline.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["layer"] + disc_cats)
    for l in range(N_LAYERS):
        w.writerow([l] + [f"{cit_disc[l, i]:.6f}" for i in range(len(disc_cats))])

# CIT scenario CSV
with open(os.path.join(OUTPUT_DIR, "cit_scenario.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["layer"] + scen_cats)
    for l in range(N_LAYERS):
        w.writerow([l] + [f"{cit_scen[l, i]:.6f}" for i in range(len(scen_cats))])

# Profile summary CSV
with open(os.path.join(OUTPUT_DIR, "profile_summary.csv"), "w") as f:
    w = csv.writer(f)
    w.writerow(["Profile", "Description", "N_Retained", "N_Pruned", "Compressed_Params", "Total_Params", "PRR"])
    for pname in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]:
        r = profile_results[pname]
        w.writerow([pname, r["description"], r["n_retained"], r["n_pruned"], 
                    r["compressed_params"], r["total_params"], f"{r['PRR']:.4f}"])

print(f"\n{'=' * 60}")
print(f"PHASE 1 COMPLETE")
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"  - cit_results.json (full results)")
print(f"  - cit_language.csv")
print(f"  - cit_discipline.csv")
print(f"  - cit_scenario.csv")
print(f"  - profile_summary.csv")
print(f"\nKey findings:")
print(f"  Cross-axis mean r = {r_cross_mean:.4f}")
print(f"  Lang-Disc cross r  = {r_lang_disc:.4f}")
print(f"  Capability Cliff (lang):  {lang_cliff.mean():.2f}x")
print(f"  Capability Cliff (disc):  {disc_cliff.mean():.2f}x")
print(f"  Capability Cliff (scen):  {scen_cliff.mean():.2f}x")
print(f"{'=' * 60}")
