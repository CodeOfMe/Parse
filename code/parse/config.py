"""
Configuration dataclasses and 12 preservation profiles (P1-P12) for PARSE.

Profiles define which (Language x Discipline x Scenario) combinations the user
wants to preserve. Each profile maps to the paper's Table 1.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


LANGUAGES = ["zh", "en", "ja", "fr", "de", "ru", "es", "ko"]
DISCIPLINES = ["math", "physics", "logic", "history", "geography", "literature"]
SCENARIOS = ["fc", "code", "math_reasoning", "translation", "chat"]

AXIS_NAMES = {
    "lang": LANGUAGES,
    "disc": DISCIPLINES,
    "scen": SCENARIOS,
}


@dataclass
class PARSEConfig:
    """Top-level configuration for a PARSE experiment run."""

    # Model
    base_model_path: str
    device: str = "cuda"

    # CIT computation
    calibration_samples_per_category: int = 15
    cit_alpha: float = 0.6  # balance activation (α) vs gradient (1-α) in CIT

    # Pruning
    target_sparsity: float = 0.5
    n_layers_total: int = 24

    # Preservation profile
    languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    disciplines: List[str] = field(default_factory=lambda: ["math", "logic"])
    scenarios: List[str] = field(default_factory=lambda: ["fc", "math_reasoning"])

    # DCR (Dynamic Capability Router)
    dcr_hidden_dim: Optional[int] = None  # auto-computed as hidden_size // 4
    enable_dcr: bool = True

    # Flywheel recovery
    flywheel_rounds: int = 3
    flywheel_samples_per_round: int = 64
    flywheel_learning_rate: float = 1e-5
    flywheel_batch_size: int = 4
    enable_grpo: bool = True
    grpo_group_size: int = 4

    # Export
    export_dir: str = "export"
    gguf_quantization: str = "Q4_K_M"

    # Output
    output_dir: str = "results/experiments"
    save_checkpoints: bool = True


@dataclass
class PreservationProfile:
    """A specific (Language x Discipline x Scenario) preservation profile."""

    name: str
    languages: List[str]
    disciplines: List[str]
    scenarios: List[str]
    description: str = ""

    @property
    def n_capabilities(self) -> int:
        return len(self.languages) * len(self.disciplines) * len(self.scenarios)

    @property
    def n_preserved(self) -> int:
        """Number of preserved capability dimensions in the profile."""
        return self.n_capabilities


# ---- 12 Preservation Profiles (Paper Table 1) ----

PROFILES: Dict[str, PreservationProfile] = {
    "P1": PreservationProfile(
        "P1",
        ["zh", "en"],
        ["math", "logic"],
        ["fc", "math_reasoning"],
        "Chinese + English STEM + Agent",
    ),
    "P2": PreservationProfile(
        "P2",
        ["zh", "en", "ja"],
        ["math", "physics"],
        ["fc", "code", "math_reasoning", "translation", "chat"],
        "East Asian + STEM",
    ),
    "P3": PreservationProfile(
        "P3",
        ["en"],
        ["math"],
        ["fc", "code", "math_reasoning", "translation", "chat"],
        "English math specialist",
    ),
    "P4": PreservationProfile(
        "P4",
        ["zh"],
        ["math", "physics", "logic", "history", "geography", "literature"],
        ["fc", "code", "math_reasoning", "translation", "chat"],
        "Chinese full-capability",
    ),
    "P5": PreservationProfile(
        "P5",
        ["zh", "en", "ja", "fr", "de", "ru", "es", "ko"],
        ["math", "logic", "physics"],
        ["fc"],
        "Multilingual STEM agent",
    ),
    "P6": PreservationProfile(
        "P6",
        ["zh", "en"],
        ["math", "physics", "logic", "history", "geography", "literature"],
        ["fc", "code"],
        "Bilingual developer agent",
    ),
    "P7": PreservationProfile(
        "P7",
        ["zh", "en", "ja", "fr", "de", "ru", "es", "ko"],
        ["math"],
        ["math_reasoning"],
        "Multilingual math solver",
    ),
    "P8": PreservationProfile(
        "P8",
        ["zh", "en", "ja", "fr"],
        ["math", "physics", "logic", "history", "geography", "literature"],
        ["translation"],
        "Quad-lingual translator",
    ),
    "P9": PreservationProfile(
        "P9",
        ["zh", "en", "ja", "fr", "de", "ru", "es", "ko"],
        ["math", "physics", "logic", "history", "geography", "literature"],
        ["fc"],
        "Universal function caller",
    ),
    "P10": PreservationProfile(
        "P10",
        ["zh", "en"],
        ["math", "physics", "logic", "history", "geography", "literature"],
        ["fc", "code", "math_reasoning", "translation", "chat"],
        "Bilingual full-capability",
    ),
    "P11": PreservationProfile(
        "P11",
        ["zh", "en", "ja", "fr", "de", "ru", "es", "ko"],
        ["math", "logic"],
        ["fc", "code", "math_reasoning", "translation", "chat"],
        "Universal STEM preservation",
    ),
    "P12": PreservationProfile(
        "P12",
        ["zh", "en"],
        ["math", "logic", "physics"],
        ["fc", "code", "math_reasoning"],
        "Full targeted preservation",
    ),
}


def get_profile(name: str) -> PreservationProfile:
    """Retrieve a preservation profile by name (P1-P12)."""
    if name not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


def profile_to_config(profile: PreservationProfile, **overrides) -> PARSEConfig:
    """Convert a PreservationProfile to a PARSEConfig with optional overrides."""
    return PARSEConfig(
        languages=profile.languages,
        disciplines=profile.disciplines,
        scenarios=profile.scenarios,
        **overrides,
    )
