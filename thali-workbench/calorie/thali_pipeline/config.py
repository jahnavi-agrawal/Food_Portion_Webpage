"""Loads configs/weights.yaml into a plain dict, resolving every relative
path against the repo root so scripts can be run from anywhere.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(value: str) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return str(p)


def _resolve_paths_in_place(node: Any, path_keys) -> None:
    """Recursively resolve any dict value whose key is in `path_keys`."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in path_keys and isinstance(v, str):
                node[k] = _resolve_path(v)
            else:
                _resolve_paths_in_place(v, path_keys)
    elif isinstance(node, list):
        for item in node:
            _resolve_paths_in_place(item, path_keys)


_PATH_KEYS = {
    "weights",
    "labels_path",
    "base_checkpoint",
    "finetuned_checkpoint",
    "dir",
    "path",
    "directory",
}

# NOTE: sam2.model_cfg is deliberately NOT resolved to an absolute path. It's
# a Hydra config identifier resolved by the installed `sam2` pip package's
# own internal config search path (e.g. "configs/sam2.1/sam2.1_hiera_t.yaml"
# inside the sam2 package), not a file that lives in this repo.


def load_config(config_path: str = "configs/weights.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _resolve_paths_in_place(cfg, _PATH_KEYS)
    return cfg
