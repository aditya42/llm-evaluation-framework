from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
