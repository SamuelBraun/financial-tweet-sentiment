"""
Text Mining Group XX — Financial Tweet Sentiment Classification.

This package contains reusable modules imported by the experimentation
and final pipeline notebooks.
"""

import os
from pathlib import Path

# Project root is the group_XX/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory — check inside group_XX first, fall back to repo root
DATA_DIR = PROJECT_ROOT / "data"
if not (DATA_DIR / "train.csv").exists():
    DATA_DIR = PROJECT_ROOT.parent / "data"

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for d in [FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42
