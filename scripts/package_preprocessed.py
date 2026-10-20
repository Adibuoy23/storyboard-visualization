#!/usr/bin/env python3
"""
package_preprocessed.py
=======================
Bundle the per-participant preprocessed CSVs for each dataset into a zip file
under js-app/public/downloads/, so users of the web app can download them.

Which files go into which bundle is defined in scripts/preprocessed_manifest.json.
Each zip also gets a README.txt listing the files, their descriptions and columns.

Usage
-----
  python3 scripts/package_preprocessed.py            # all datasets
  python3 scripts/package_preprocessed.py mrbean     # one dataset key
"""

import csv, json, os, sys, zipfile
from datetime import date

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "scripts", "preprocessed_manifest.json")
OUT_DIR  = os.path.join(ROOT, "js-app", "public", "downloads")


def resolve(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def header_of(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def package(key, spec):
    print(f"[{key}] {spec['label']}")
    lines = [f"{spec['label']} - preprocessed per-participant data",
             f"Packaged {date.today().isoformat()} from https://github.com/Adibuoy23/storyboard-visualization",
             ""]
    out = os.path.join(OUT_DIR, spec["zip"])
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in spec["files"]:
            src = resolve(f["src"])
            if not os.path.exists(src):
                print(f"  MISSING: {src}")
                return False
            z.write(src, f["name"])
            print(f"  + {f['name']}  ({os.path.getsize(src)/1e6:.1f} MB)")
            lines += [f["name"], f"  {f['desc']}", f"  columns: {', '.join(header_of(src))}", ""]
        z.writestr("README.txt", "\n".join(lines))
    print(f"  -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return True


def main():
    manifest = json.load(open(MANIFEST))
    keys = sys.argv[1:] or [k for k in manifest if not k.startswith("_")]
    os.makedirs(OUT_DIR, exist_ok=True)
    ok = all(package(k, manifest[k]) for k in keys)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
