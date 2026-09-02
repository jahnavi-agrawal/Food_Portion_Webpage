"""Device resolution shared by every model wrapper."""

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device string ("auto" | "cpu" | "cuda" | "mps") to a torch.device.

    "auto" picks cuda > mps > cpu, matching the pattern used throughout the
    original repo's predict.py / predict_thali.py scripts.
    """
    preference = (preference or "auto").lower()

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device: 'cuda' requested but not available")
        return torch.device("cuda")
    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("device: 'mps' requested but not available")
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
