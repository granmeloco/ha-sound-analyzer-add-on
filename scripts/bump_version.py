#!/usr/bin/env python3
import re
import argparse
from pathlib import Path


def read_version(text: str):
    m = re.search(r"^version:\s*\"?([0-9]+)\.([0-9]+)\.([0-9]+)\"?", text, flags=re.M)
    if not m:
        raise SystemExit("Unable to find version in config.yaml")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def write_version(text: str, major: int, minor: int, patch: int):
    new_text = re.sub(r"^version:\s*\"?[0-9]+\.[0-9]+\.[0-9]+\"?",
                      f"version: \"{major}.{minor}.{patch}\"",
                      text, count=1, flags=re.M)
    return new_text


def bump(path: Path, kind: str):
    text = path.read_text(encoding="utf-8")
    major, minor, patch = read_version(text)
    if kind == "patch":
        patch += 1
    elif kind == "minor":
        minor += 1
        patch = 0
    elif kind == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise SystemExit("Unknown bump kind")
    new_text = write_version(text, major, minor, patch)
    path.write_text(new_text, encoding="utf-8")
    print(f"Bumped version to {major}.{minor}.{patch}")
    return f"{major}.{minor}.{patch}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bump", choices=["patch", "minor", "major"], default="patch")
    p.add_argument("--file", default="config.yaml")
    args = p.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    bump(path, args.bump)


if __name__ == '__main__':
    main()
