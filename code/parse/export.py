"""
GGUF model export for PARSE-compressed models.

Exports a HuggingFace model to GGUF format, compatible with:
- llama.cpp (inference engine)
- MoXing (OpenAI API-compatible server at /Users/fred/Documents/GitHub/cycleuser/MoXing)

Pipeline:
    1. Save model in HuggingFace format (safetensors + config.json)
    2. Convert to FP16 GGUF via llama.cpp's convert_hf_to_gguf.py
    3. Quantize to target format (Q4_K_M default) via llama-quantize
    4. Copy quantized GGUF to MoXing models directory

Requirements:
    - llama.cpp cloned and built (bundled with MoXing or standalone)
    - MoXing installed for serving (optional, for verification)
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict


# ── Paths ────────────────────────────────────────────────────────────
# Searched in order; first match wins. Set LLAMA_CPP_PATH and MOXING_PATH
# environment variables to override.

LLAMA_CPP_DIRS = [
    Path(p) for p in [
        os.environ.get("LLAMA_CPP_PATH", ""),
        (Path.home() / "llama.cpp").as_posix(),
        "/usr/local/share/llama.cpp",
    ] if p
]

MOXING_DIR = Path(os.environ.get("MOXING_PATH", Path.home() / "MoXing"))
MOXING_MODELS = MOXING_DIR / "models"


def _find_llama_cpp() -> Path:
    """Locate llama.cpp installation (bundled with MoXing or standalone)."""
    for d in LLAMA_CPP_DIRS:
        if d.exists() and (d / "convert_hf_to_gguf.py").exists():
            return d
    raise FileNotFoundError(
        "llama.cpp not found. Clone it:\n"
        "  git clone https://github.com/ggml-org/llama.cpp.git\n"
        f"Checked: {[str(d) for d in LLAMA_CPP_DIRS]}"
    )


def _find_llama_quantize(llama_dir: Path) -> Path:
    """Find llama-quantize binary (system PATH or llama.cpp build dir)."""
    # Check PATH
    which = shutil.which("llama-quantize")
    if which:
        return Path(which)

    # Check llama.cpp build directory
    for sub in ["build/bin", "build"]:
        p = llama_dir / sub / "llama-quantize"
        if p.exists():
            return p

    raise FileNotFoundError(
        "llama-quantize not found. Build llama.cpp:\n"
        f"  cd {llama_dir} && mkdir build && cd build && cmake .. && make -j"
    )


def export_to_gguf(
    model,
    tokenizer,
    output_name: str = "parse_model",
    output_dir: str = "export",
    quantization: str = "Q4_K_M",
    copy_to_moxing: bool = True,
) -> Dict[str, str]:
    """
    Export a HuggingFace model to GGUF format for llama.cpp / MoXing inference.

    Parameters
    ----------
    model : nn.Module
        The HuggingFace model to export.
    tokenizer : PreTrainedTokenizer
    output_name : str
        Base name for output files.
    output_dir : str
        Directory for intermediate and final outputs.
    quantization : str
        llama.cpp quantization type (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)
    copy_to_moxing : bool
        If True, copy the final GGUF to MoXing's models directory.

    Returns
    -------
    Dict with paths: {"hf_dir", "fp16_gguf", "quantized_gguf", "moxing_path"}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # ── Step 1: Save HuggingFace format ──
    hf_dir = output_dir / f"{output_name}_hf"
    hf_dir.mkdir(exist_ok=True)

    model.save_pretrained(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    # Fix tokenizer_config.json for llama.cpp compatibility
    _fix_tokenizer_config(hf_dir)
    paths["hf_dir"] = str(hf_dir)
    print(f"  [1/4] HuggingFace model saved → {hf_dir}")

    # ── Step 2: Convert to FP16 GGUF ──
    llama_dir = _find_llama_cpp()
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    fp16_gguf = output_dir / f"{output_name}-F16.gguf"

    cmd = [
        sys.executable,
        str(convert_script),
        str(hf_dir),
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")

    paths["fp16_gguf"] = str(fp16_gguf)
    fp16_size = fp16_gguf.stat().st_size / (1024**2)
    print(f"  [2/4] FP16 GGUF created → {fp16_gguf} ({fp16_size:.0f} MB)")

    # ── Step 3: Quantize ──
    quantize_bin = _find_llama_quantize(llama_dir)
    quant_gguf = output_dir / f"{output_name}-{quantization}.gguf"

    cmd = [
        str(quantize_bin),
        str(fp16_gguf),
        str(quant_gguf),
        quantization,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed:\n{result.stderr}")

    paths["quantized_gguf"] = str(quant_gguf)
    quant_size = quant_gguf.stat().st_size / (1024**2)
    print(f"  [3/4] {quantization} GGUF created → {quant_gguf} ({quant_size:.0f} MB)")

    # ── Step 4: Copy to MoXing ──
    if copy_to_moxing and MOXING_MODELS.parent.exists():
        MOXING_MODELS.mkdir(parents=True, exist_ok=True)
        dest = MOXING_MODELS / quant_gguf.name
        shutil.copy2(str(quant_gguf), str(dest))
        paths["moxing_path"] = str(dest)
        print(f"  [4/4] Copied to MoXing → {dest}")
        print(f"\n  Serve with: moxing serve {dest}")
    else:
        print(f"  [4/4] MoXing copy skipped (MoXing not found at {MOXING_DIR})")

    _save_export_manifest(output_dir, output_name, paths)
    return paths


def _fix_tokenizer_config(hf_dir: Path):
    """Fix tokenizer_config.json for llama.cpp GGUF conversion compatibility."""
    tok_config_path = hf_dir / "tokenizer_config.json"
    if not tok_config_path.exists():
        return

    with open(tok_config_path, "r") as f:
        config = json.load(f)

    # llama.cpp expects these fields
    changed = False
    for key in ["add_bos_token", "add_eos_token"]:
        if key not in config:
            config[key] = True
            changed = True

    if "chat_template" in config:
        config["chat_template"] = config["chat_template"].replace("\\n", "\n")
        changed = True

    if changed:
        with open(tok_config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def _save_export_manifest(output_dir: Path, name: str, paths: Dict[str, str]):
    """Save export metadata for reproducibility."""
    manifest = {
        "model_name": name,
        "export_paths": paths,
        "size_mb": {
            "fp16": (
                Path(paths["fp16_gguf"]).stat().st_size / (1024**2)
                if Path(paths["fp16_gguf"]).exists()
                else 0
            ),
            "quantized": (
                Path(paths.get("quantized_gguf", "")).stat().st_size / (1024**2)
                if paths.get("quantized_gguf") and Path(paths["quantized_gguf"]).exists()
                else 0
            ),
        },
    }
    manifest_path = output_dir / f"{name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
