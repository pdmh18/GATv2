from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]


def project_path(path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def load_config(path: str = 'configs/config.yaml') -> dict:
    config_path = project_path(path)
    with config_path.open(encoding='utf-8') as stream:
        if config_path.suffix == '.json':
            config = json.load(stream)
        else:
            # Legacy training config retains .env loading and YAML dependencies.
            import yaml
            from dotenv import load_dotenv
            load_dotenv(ROOT / '.env')
            config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('Config must be an object')
    return config
