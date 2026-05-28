#!/usr/bin/env python3
"""
Extract VQA questions and descriptions from em_data / mada_data / sama_data,
filtered to only (image, mat_id) pairs present in our release JSONs.
Train/val split is dropped (merged).

Key mappings:
  synmat: VQA key = basename of filepath (e.g. "AI09_002_frame0780_...exr")
  realmat: VQA key = filepath with "/realmat/" prefix stripped (e.g. "material_20241203/000625.jpg")
  sama:    VQA key = basename of filepath (e.g. "video16_frame0.exr")
"""
import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.paths import data_root, material_data_root


def load_release_lookup(json_path, key_fn):
    """Return {vqa_key: set(mat_id_str)} using key_fn(filepath) as the lookup key."""
    with open(json_path) as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    lookup = {}
    for s in samples:
        k = key_fn(s["filepath"])
        mid = str(s["mat_id"])
        lookup.setdefault(k, set()).add(mid)
    return lookup


def merge_jsons(*paths, key="samples"):
    """Load and merge JSONs, unwrapping 'samples' wrapper if present."""
    merged = {}
    for p in paths:
        with open(p) as f:
            raw = json.load(f)
        data = raw[key] if isinstance(raw, dict) and key in raw else raw
        for k, v in data.items():
            if k not in merged:
                merged[k] = v
            elif isinstance(v, dict) and isinstance(merged[k], dict):
                merged[k].update(v)
    return merged


def filter_to_retained(data, lookup):
    """Keep only (file, mat_id) pairs present in lookup."""
    out = {}
    for key, mats in data.items():
        if key not in lookup:
            continue
        retained_mids = lookup[key]
        filtered = {mid: v for mid, v in mats.items() if mid in retained_mids}
        if filtered:
            out[key] = filtered
    return out


def extract_dataset(name, release_json, key_fn, vqa_paths, desc_paths, out_vqa, out_desc):
    print(f"\n=== {name} ===")
    lookup = load_release_lookup(release_json, key_fn)
    print(f"  Retained unique files: {len(lookup)}")

    vqa_all = merge_jsons(*vqa_paths, key="samples")
    print(f"  VQA total files before filter: {len(vqa_all)}")
    vqa_filtered = filter_to_retained(vqa_all, lookup)
    print(f"  VQA files after filter: {len(vqa_filtered)}")
    print(f"  VQA (file, mat_id) pairs: {sum(len(v) for v in vqa_filtered.values())}")

    desc_all = merge_jsons(*desc_paths)
    print(f"  Desc total files before filter: {len(desc_all)}")
    desc_filtered = filter_to_retained(desc_all, lookup)
    print(f"  Desc files after filter: {len(desc_filtered)}")
    print(f"  Desc (file, mat_id) pairs: {sum(len(v) for v in desc_filtered.values())}")

    with open(out_vqa, "w") as f:
        json.dump(vqa_filtered, f)
    with open(out_desc, "w") as f:
        json.dump(desc_filtered, f)
    print(f"  Saved -> {out_vqa}")
    print(f"  Saved -> {out_desc}")

    vqa_pairs = {(k, mid) for k, mats in vqa_filtered.items() for mid in mats}
    desc_pairs = {(k, mid) for k, mats in desc_filtered.items() for mid in mats}
    only_vqa = vqa_pairs - desc_pairs
    only_desc = desc_pairs - vqa_pairs
    if only_vqa or only_desc:
        print(f"  WARNING: {len(only_vqa)} pairs only in VQA, {len(only_desc)} only in desc")
    else:
        print("  OK: VQA and desc have identical (file, mat_id) pairs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract retained material VQA questions and descriptions."
    )
    parser.add_argument(
        "--source-dir",
        default=str(data_root() / "tmp_maoam_data"),
        help="Directory containing em_data, mada_data, and sama_data",
    )
    parser.add_argument(
        "--release-dir",
        default=str(material_data_root()),
        help="Directory containing the *_release.json files",
    )
    parser.add_argument(
        "--out-dir",
        default=str(material_data_root()),
        help="Directory for the extracted description and VQA JSON files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = os.path.expanduser(args.source_dir)
    release_dir = os.path.expanduser(args.release_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- synmat ---
    extract_dataset(
        name="synmat",
        release_json=os.path.join(release_dir, "synmat_release.json"),
        key_fn=os.path.basename,  # /synmat/FILE.exr -> FILE.exr
        vqa_paths=[
            os.path.join(source_dir, "em_data", "em_vqa_questions_train.json"),
            os.path.join(source_dir, "em_data", "em_vqa_questions_val.json"),
        ],
        desc_paths=[
            os.path.join(source_dir, "em_data", "232b_som_filtered", "em_mat_loc_desc_train.json"),
            os.path.join(source_dir, "em_data", "232b_som_filtered", "em_mat_loc_desc_val.json"),
        ],
        out_vqa=os.path.join(out_dir, "synmat_vqa.json"),
        out_desc=os.path.join(out_dir, "synmat_descriptions.json"),
    )

    # --- realmat ---
    extract_dataset(
        name="realmat",
        release_json=os.path.join(release_dir, "realmat_release.json"),
        key_fn=lambda fp: fp[len("/realmat/"):] if fp.startswith("/realmat/") else fp,
        vqa_paths=[
            os.path.join(source_dir, "mada_data", "mada_vqa_questions_train.json"),
            os.path.join(source_dir, "mada_data", "mada_vqa_questions_val.json"),
        ],
        desc_paths=[
            os.path.join(source_dir, "mada_data", "232b_som_filtered", "mada_mat_loc_desc_train.json"),
            os.path.join(source_dir, "mada_data", "232b_som_filtered", "mada_mat_loc_desc_val.json"),
        ],
        out_vqa=os.path.join(out_dir, "realmat_vqa.json"),
        out_desc=os.path.join(out_dir, "realmat_descriptions.json"),
    )

    # --- sama ---
    extract_dataset(
        name="sama",
        release_json=os.path.join(release_dir, "sama_release.json"),
        key_fn=os.path.basename,  # /sama/video16_frame0.exr -> video16_frame0.exr
        vqa_paths=[
            os.path.join(source_dir, "sama_data", "sama_vqa_questions_train.json"),
            os.path.join(source_dir, "sama_data", "sama_vqa_questions_val.json"),
        ],
        desc_paths=[
            os.path.join(source_dir, "sama_data", "232b_som_filtered", "sama_mat_loc_desc_train.json"),
            os.path.join(source_dir, "sama_data", "232b_som_filtered", "sama_mat_loc_desc_val.json"),
        ],
        out_vqa=os.path.join(out_dir, "sama_vqa.json"),
        out_desc=os.path.join(out_dir, "sama_descriptions.json"),
    )


if __name__ == "__main__":
    main()
