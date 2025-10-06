"""Configuration helpers for the hist CLI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class RunConfig:
    command: str
    options: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        if yaml is None:
            path.write_text(json.dumps(data, indent=2))
        else:
            with path.open("w", encoding="utf8") as handle:
                yaml.safe_dump(data, handle, sort_keys=False)


def load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if yaml is None:
        return json.loads(path.read_text())
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle) or {}


def merge_cli_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value format")
        key, value = item.split("=", 1)
        if yaml is None:
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value
        else:
            result[key] = yaml.safe_load(value)
    return result


__all__ = ["RunConfig", "load_yaml_config", "merge_cli_overrides"]
