"""Add or replace one model in the speed-limit model catalog published on Hugging Face.

Usage: python -m nnslr_tools.model_catalog <speed_models.json> --name <nome> --run-id <id>
           --model-dir <stage>/models/<nome> --base-url https://huggingface.co/<repo>/resolve/main

Called by scripts/publish_model.sh after the bundle passed "nnslr verify-bundle". A missing catalog
file starts a new catalog; the model with the same name is replaced in place, keeping its position
in the list. Standard library only: it runs without any dependency of this repo.

Catalog format, pretty JSON with the keys always in this order:

    {
      "schema_version": 1,
      "models": [
        {
          "name": "baseline-001",
          "display_date": "October 21, 2026",
          "run_id": "baseline-001",
          "created": "2026-10-21T08:15:00Z",
          "bundle": {
            "schema_version": 1,
            "feature_id": "nnslr-speed-vision",
            "digest": "<sha256>",
            "dataset_sha256": "<sha256>",
            "split_sha256": "<sha256>",
            "advisory_only": true
          },
          "files": [
            {
              "path": "models/baseline-001/capability.json",
              "size": 412,
              "sha256": "<sha256>",
              "url": "https://huggingface.co/<repo>/resolve/main/models/baseline-001/capability.json"
            }
          ]
        }
      ]
    }

- "created" is UTC; "display_date" is the same day written for people.
- "bundle" comes from the bundle's config.json and capability.json; "digest" is the sha256 of its
  sha256sums.json, the same bundle_digest that "nnslr verify-bundle" prints.
- "files" lists every file of the model folder sorted by name. "path" is relative to the repo root
  and "url" is <base url>/<path>. The sha256 is taken from the bundle's own sha256sums.json; only
  sha256sums.json itself, which cannot list its own hash, is hashed here.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re


CATALOG_SCHEMA_VERSION = 1
MANIFEST_NAME = "sha256sums.json"
CONFIG_NAME = "config.json"
CAPABILITY_NAME = "capability.json"
# Letters, digits, _ . - starting with a letter or a digit: never "." or "..", never an option.
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def is_valid_name(name: str) -> bool:
    if re.fullmatch(NAME_PATTERN, name):
        return True
    return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"manca {path}")
    # A download can answer 200 with an empty or HTML page: stop with a message, not a traceback.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise SystemExit(f"JSON non valido: {path}")
    if not isinstance(payload, dict):
        raise SystemExit(f"non e' un oggetto JSON: {path}")
    return payload


def make_bundle_info(model_dir: Path) -> dict:
    config = read_json_object(model_dir / CONFIG_NAME)
    capability = read_json_object(model_dir / CAPABILITY_NAME)
    return {
        "schema_version": config.get("schema_version"),
        "feature_id": config.get("feature_id"),
        "digest": file_sha256(model_dir / MANIFEST_NAME),
        "dataset_sha256": config.get("dataset_sha256"),
        "split_sha256": config.get("split_sha256"),
        "advisory_only": capability.get("advisory_only"),
    }


def make_files(name: str, model_dir: Path, base_url: str) -> list[dict]:
    manifest = read_json_object(model_dir / MANIFEST_NAME)
    known_hashes = manifest.get("files")
    if not isinstance(known_hashes, dict):
        raise SystemExit(f"{model_dir / MANIFEST_NAME} non ha l'elenco \"files\"")

    files = []
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            raise SystemExit(f"nella cartella del modello c'e' qualcosa che non e' un file: {path}")
        if path.name in known_hashes:
            sha256 = known_hashes[path.name]
        else:
            sha256 = file_sha256(path)
        relative_path = f"models/{name}/{path.name}"
        files.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": sha256,
                "url": f"{base_url}/{relative_path}",
            }
        )
    return files


def make_entry(name: str, run_id: str, model_dir: Path, base_url: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    display_date = f"{MONTHS[now.month - 1]} {now.day}, {now.year}"
    return {
        "name": name,
        "display_date": display_date,
        "run_id": run_id,
        "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle": make_bundle_info(model_dir),
        "files": make_files(name, model_dir, base_url),
    }


def load_catalog(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": CATALOG_SCHEMA_VERSION, "models": []}
    catalog = read_json_object(path)
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise SystemExit(f"catalogo con schema_version diversa da {CATALOG_SCHEMA_VERSION}: {path}")
    models = catalog.get("models")
    if not isinstance(models, list):
        raise SystemExit(f"catalogo senza l'elenco \"models\": {path}")
    for entry in models:
        if not isinstance(entry, dict):
            raise SystemExit(f"catalogo con un modello che non e' un oggetto JSON: {path}")
    return {"schema_version": CATALOG_SCHEMA_VERSION, "models": models}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", type=Path, help="speed_models.json to update (created if missing)")
    parser.add_argument("--name", required=True, help="model name: letters, digits, _ . -")
    parser.add_argument("--run-id", required=True, help="training run the bundle comes from")
    parser.add_argument("--model-dir", type=Path, required=True, help="staged folder <stage>/models/<name>")
    parser.add_argument("--base-url", required=True, help="https://huggingface.co/<repo>/resolve/main")
    args = parser.parse_args(argv)

    if not is_valid_name(args.name):
        raise SystemExit(f"nome non valido: {args.name} (solo lettere, numeri, _ . - e all'inizio una lettera o un numero)")
    if args.run_id == "":
        raise SystemExit("run id vuoto")
    if not args.model_dir.is_dir():
        raise SystemExit(f"{args.model_dir} non e' una cartella")
    # The paths in the catalog are models/<name>/...: the staged folder must be exactly that one.
    if args.model_dir.name != args.name:
        raise SystemExit(f"la cartella {args.model_dir} non si chiama {args.name}")
    base_url = args.base_url.rstrip("/")

    catalog = load_catalog(args.catalog)
    entry = make_entry(args.name, args.run_id, args.model_dir, base_url)

    position = None
    for number, model in enumerate(catalog["models"]):
        if model.get("name") == args.name:
            position = number

    if position is None:
        catalog["models"].append(entry)
        print(f"catalogo: aggiunto {args.name} ({len(entry['files'])} file, bundle {entry['bundle']['digest']})")
    else:
        catalog["models"][position] = entry
        print(f"catalogo: sostituito {args.name} ({len(entry['files'])} file, bundle {entry['bundle']['digest']})")

    args.catalog.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
