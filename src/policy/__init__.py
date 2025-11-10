# src/policy/__init__.py

# Re-export the legacy rule-based Policy class (from src/legacy_policy.py)
from ..legacy_policy import Policy

# Re-export the SFT controller (from src/policy/sft_controller.py)
from .sft_controller import SFTPolicyController

__all__ = ["Policy", "SFTPolicyController"]