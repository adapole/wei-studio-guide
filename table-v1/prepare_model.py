"""Prepare a private model tree with the exact XML bytes used for calibration."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

source, destination = map(Path, sys.argv[1:])
pins = json.loads(Path(__file__).with_name("xml-sha256.json").read_text())
package = Path("src/orca_sim")
for item in (source / package).rglob("*"):
    if not item.is_file() or "__pycache__" in item.parts:
        continue
    name = item.relative_to(source / package)
    target = destination / package / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if name.as_posix() in pins:
        lf = item.read_bytes().replace(b"\r\n", b"\n")
        for content in (lf, lf.replace(b"\n", b"\r\n")):
            if hashlib.sha256(content).hexdigest() == pins[name.as_posix()]:
                target.write_bytes(content)
                break
        else:
            raise ValueError(f"Upstream XML differs from calibration: {name}")
    elif item.suffix == ".py":
        shutil.copyfile(item, target)
    else:
        target.symlink_to(item.resolve())
