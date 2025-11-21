from pathlib import Path
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_config(
    config_path: str | Path,
    overrides: Sequence[str] | None = None,
    **override_kwargs: object,
) -> DictConfig:
    """Load a Hydra config from a file path with optional override kwargs."""
    path = Path(config_path).resolve()
    override_args = list(overrides) if overrides else []
    for key, value in override_kwargs.items():
        override_args.append(f"{key}={value}")

    with initialize_config_dir(
        config_dir=str(path.parent),
        job_name="compose_config",
        version_base=None,
    ):
        return compose(config_name=path.stem, overrides=override_args)


def extract_target_config(cfg: DictConfig, *, context: str) -> Any:
    """
    Return the portion of a config that contains a Hydra _target_.
    - If the root has _target_, return root.
    - If the root has exactly one key and its value has _target_, return that value.
    """
    if "_target_" in cfg:
        return cfg
    keys = list(cfg.keys())
    if len(keys) == 1:
        sub_cfg = cfg[keys[0]]
        if isinstance(sub_cfg, DictConfig | dict) and "_target_" in sub_cfg:
            return sub_cfg
    raise KeyError(f"No _target_ found in config for {context}")
