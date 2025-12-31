#!/usr/bin/env python3
import yaml
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "wp_audio_trigger" / "config.yaml"
REPO_JSON_PATH = Path(__file__).resolve().parent.parent / "repository.json"

# Read version from config.yaml
def get_config_version():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return str(data.get("version", "0.0.0"))

# Update version in repository.json
def sync_repo_json(version):
    with open(REPO_JSON_PATH, "r", encoding="utf-8") as f:
        repo = json.load(f)
    # Find the first add-on and update its version
    for addon in repo.get("addons", {}).values():
        addon["version"] = version
    with open(REPO_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(repo, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    version = get_config_version()
    sync_repo_json(version)
