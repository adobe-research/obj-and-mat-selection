"""
Convert jxu124/{refcoco,refcocoplus,refcocog} parquet files to the original
REFER API format:
  {out_dir}/{dataset}/refs({splitBy}).p   -- pickle of list[ref_dict]
  {out_dir}/{dataset}/instances.json      -- COCO-style JSON

Run once:
  python tools/convert_refcoco_parquet.py
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from utils.paths import data_root

DATASETS = {
    "refcoco": {
        "hf_repo": "jxu124/refcoco",
        "split_by": "unc",
        "splits": [
            "data/train-00000-of-00001-94431d5f4bd5b93f.parquet",
            "data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet",
            "data/test-00000-of-00001-82af0c1b600890ac.parquet",
            "data/testB-00000-of-00001-60990e4598892dc1.parquet",
        ],
    },
    "refcoco+": {
        "hf_repo": "jxu124/refcocoplus",
        "split_by": "unc",
        "splits": [
            "data/train-00000-of-00001-7294665695c630ee.parquet",
            "data/validation-00000-of-00001-8c57d66282bc60c9.parquet",
            "data/test-00000-of-00001-2b8e5d26906553b9.parquet",
            "data/testB-00000-of-00001-4f1178d399f1874a.parquet",
        ],
    },
    "refcocog": {
        "hf_repo": "jxu124/refcocog",
        "split_by": "umd",
        "splits": [
            "data/train-00000-of-00001-4fe3e6340cfb69ed.parquet",
            "data/validation-00000-of-00001-15168dfe7b5961e5.parquet",
            "data/test-00000-of-00001-2316f36b19cd7f72.parquet",
        ],
    },
}

# COCO categories needed by the REFER API (standard 80 COCO classes)
COCO_CATEGORIES = [
    {"supercategory": "person", "id": 1, "name": "person"},
    {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
    {"supercategory": "vehicle", "id": 3, "name": "car"},
    {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
    {"supercategory": "vehicle", "id": 5, "name": "airplane"},
    {"supercategory": "vehicle", "id": 6, "name": "bus"},
    {"supercategory": "vehicle", "id": 7, "name": "train"},
    {"supercategory": "vehicle", "id": 8, "name": "truck"},
    {"supercategory": "vehicle", "id": 9, "name": "boat"},
    {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
    {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
    {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
    {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
    {"supercategory": "outdoor", "id": 15, "name": "bench"},
    {"supercategory": "animal", "id": 16, "name": "bird"},
    {"supercategory": "animal", "id": 17, "name": "cat"},
    {"supercategory": "animal", "id": 18, "name": "dog"},
    {"supercategory": "animal", "id": 19, "name": "horse"},
    {"supercategory": "animal", "id": 20, "name": "sheep"},
    {"supercategory": "animal", "id": 21, "name": "cow"},
    {"supercategory": "animal", "id": 22, "name": "elephant"},
    {"supercategory": "animal", "id": 23, "name": "bear"},
    {"supercategory": "animal", "id": 24, "name": "zebra"},
    {"supercategory": "animal", "id": 25, "name": "giraffe"},
    {"supercategory": "accessory", "id": 27, "name": "backpack"},
    {"supercategory": "accessory", "id": 28, "name": "umbrella"},
    {"supercategory": "accessory", "id": 31, "name": "handbag"},
    {"supercategory": "accessory", "id": 32, "name": "tie"},
    {"supercategory": "accessory", "id": 33, "name": "suitcase"},
    {"supercategory": "sports", "id": 34, "name": "frisbee"},
    {"supercategory": "sports", "id": 35, "name": "skis"},
    {"supercategory": "sports", "id": 36, "name": "snowboard"},
    {"supercategory": "sports", "id": 37, "name": "sports ball"},
    {"supercategory": "sports", "id": 38, "name": "kite"},
    {"supercategory": "sports", "id": 39, "name": "baseball bat"},
    {"supercategory": "sports", "id": 40, "name": "baseball glove"},
    {"supercategory": "sports", "id": 41, "name": "skateboard"},
    {"supercategory": "sports", "id": 42, "name": "surfboard"},
    {"supercategory": "sports", "id": 43, "name": "tennis racket"},
    {"supercategory": "kitchen", "id": 44, "name": "bottle"},
    {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
    {"supercategory": "kitchen", "id": 47, "name": "cup"},
    {"supercategory": "kitchen", "id": 48, "name": "fork"},
    {"supercategory": "kitchen", "id": 49, "name": "knife"},
    {"supercategory": "kitchen", "id": 50, "name": "spoon"},
    {"supercategory": "kitchen", "id": 51, "name": "bowl"},
    {"supercategory": "food", "id": 52, "name": "banana"},
    {"supercategory": "food", "id": 53, "name": "apple"},
    {"supercategory": "food", "id": 54, "name": "sandwich"},
    {"supercategory": "food", "id": 55, "name": "orange"},
    {"supercategory": "food", "id": 56, "name": "broccoli"},
    {"supercategory": "food", "id": 57, "name": "carrot"},
    {"supercategory": "food", "id": 58, "name": "hot dog"},
    {"supercategory": "food", "id": 59, "name": "pizza"},
    {"supercategory": "food", "id": 60, "name": "donut"},
    {"supercategory": "food", "id": 61, "name": "cake"},
    {"supercategory": "furniture", "id": 62, "name": "chair"},
    {"supercategory": "furniture", "id": 63, "name": "couch"},
    {"supercategory": "furniture", "id": 64, "name": "potted plant"},
    {"supercategory": "furniture", "id": 65, "name": "bed"},
    {"supercategory": "furniture", "id": 67, "name": "dining table"},
    {"supercategory": "furniture", "id": 70, "name": "toilet"},
    {"supercategory": "electronic", "id": 72, "name": "tv"},
    {"supercategory": "electronic", "id": 73, "name": "laptop"},
    {"supercategory": "electronic", "id": 74, "name": "mouse"},
    {"supercategory": "electronic", "id": 75, "name": "remote"},
    {"supercategory": "electronic", "id": 76, "name": "keyboard"},
    {"supercategory": "electronic", "id": 77, "name": "cell phone"},
    {"supercategory": "appliance", "id": 78, "name": "microwave"},
    {"supercategory": "appliance", "id": 79, "name": "oven"},
    {"supercategory": "appliance", "id": 80, "name": "toaster"},
    {"supercategory": "appliance", "id": 81, "name": "sink"},
    {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
    {"supercategory": "indoor", "id": 84, "name": "book"},
    {"supercategory": "indoor", "id": 85, "name": "clock"},
    {"supercategory": "indoor", "id": 86, "name": "vase"},
    {"supercategory": "indoor", "id": 87, "name": "scissors"},
    {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
    {"supercategory": "indoor", "id": 89, "name": "hair drier"},
    {"supercategory": "indoor", "id": 90, "name": "toothbrush"},
]


def build_dataset(dataset_name, cfg, out_dir):
    hf_repo = cfg["hf_repo"]
    split_by = cfg["split_by"]

    refs = []
    images_by_id = {}
    anns_by_id = {}

    for parquet_file in cfg["splits"]:
        print(f"  Downloading {hf_repo}/{parquet_file}...")
        local = hf_hub_download(repo_id=hf_repo, filename=parquet_file, repo_type="dataset")
        table = pq.read_table(local)
        rows = table.to_pydict()
        n = len(rows["ref_id"])

        for i in range(n):
            ref = {
                "ref_id": int(rows["ref_id"][i]),
                "ann_id": int(rows["ann_id"][i]),
                "image_id": int(rows["image_id"][i]),
                "split": str(rows["split"][i]),
                "category_id": int(rows["category_id"][i]),
                "file_name": str(rows["file_name"][i]),
                "sent_ids": list(rows["sent_ids"][i]),
                "sentences": [
                    {
                        "raw": s["raw"],
                        "sent": s["sent"],
                        "sent_id": int(s["sent_id"]),
                        "tokens": list(s["tokens"]),
                    }
                    for s in rows["sentences"][i]
                ],
            }
            refs.append(ref)

            img_id = int(rows["image_id"][i])
            if img_id not in images_by_id:
                img_info = json.loads(rows["raw_image_info"][i])
                images_by_id[img_id] = img_info

            ann_id = int(rows["ann_id"][i])
            if ann_id not in anns_by_id:
                ann_info = json.loads(rows["raw_anns"][i])
                anns_by_id[ann_id] = ann_info

    instances = {
        "images": list(images_by_id.values()),
        "annotations": list(anns_by_id.values()),
        "categories": COCO_CATEGORIES,
    }

    out_ds_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(out_ds_dir, exist_ok=True)

    refs_path = os.path.join(out_ds_dir, f"refs({split_by}).p")
    with open(refs_path, "wb") as f:
        pickle.dump(refs, f)
    print(f"  Wrote {len(refs)} refs to {refs_path}")

    inst_path = os.path.join(out_ds_dir, "instances.json")
    with open(inst_path, "w") as f:
        json.dump(instances, f)
    print(f"  Wrote instances.json ({len(instances['images'])} imgs, {len(instances['annotations'])} anns)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=str(data_root() / "Refer_Segm"),
        help="Output directory (default: $DATA_ROOT/Refer_Segm or <repo>/data/Refer_Segm)",
    )
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()))
    args = parser.parse_args()

    for name in args.datasets:
        print(f"\n=== Converting {name} ===")
        build_dataset(name, DATASETS[name], args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
