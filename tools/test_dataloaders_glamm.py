"""
Smoke-test for all GLaMM dataloaders (no model required).
Loads 3 samples from each dataset and prints shapes.

Run from repo root:
  PYTHONPATH=. GLaMM/.venv/bin/python tools/test_dataloaders_glamm.py
"""
import os, sys, traceback
import torch

VISION_TOWER = "openai/clip-vit-large-patch14-336"
N = 3  # samples to test per dataset

_GLAMM_ROOT = os.path.join(os.path.dirname(__file__), "..", "GLaMM")
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
for _p in (_GLAMM_ROOT, _REPO_ROOT):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.paths import data_root

DATA_ROOT = str(data_root())

from model.llava import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]


def _check(name, ds, n=N):
    print(f"\n{'='*60}")
    print(f"  {name}  (len={len(ds)})")
    print(f"{'='*60}")
    ok = 0
    for i in range(min(n, len(ds))):
        try:
            item = ds[i]
            img = item.get("grounding_image")
            if img is None:
                img = item.get("image_star")
            masks = item["masks"]
            print(f"  [{i}] img={tuple(img.shape) if img is not None else None}  "
                  f"masks={tuple(masks.shape)}  "
                  f"coords={item.get('coords')}  "
                  f"classes={item.get('sampled_classes')}")
            ok += 1
        except Exception:
            print(f"  [{i}] FAILED:")
            traceback.print_exc()
    status = "PASS" if ok == min(n, len(ds)) else f"FAIL ({ok}/{min(n,len(ds))} ok)"
    print(f"  -> {status}")
    return ok == min(n, len(ds))


results = {}

# ── EntitySeg ───────────────────────────────────────────────────────────────
try:
    from dataset.entity_datasets.Entity_InsSeg import EntitySegDataset
    ds = EntitySegDataset(
        samples_json=os.path.join(DATA_ROOT, "entityseg/entityseg_insseg_val.json"),
        use_star=True,
        global_image_encoder=VISION_TOWER,
        ann_train_json=os.path.join(DATA_ROOT, "entityseg/annotations/entityseg_insseg_train_annotations.json"),
        ann_val_json=os.path.join(DATA_ROOT, "entityseg/annotations/entityseg_insseg_val_annotations.json"),
        image_roots=[
            os.path.join(DATA_ROOT, "entityseg/images/entity_01_11580/images_merge"),
            os.path.join(DATA_ROOT, "entityseg/images/entity_02_11598/images"),
            os.path.join(DATA_ROOT, "entityseg/images/entity_03_10049/images_03_10049"),
        ],
    )
    results["glamm_entityseg"] = _check("GLaMM EntitySeg", ds)
except Exception:
    print("\nGLaMM EntitySeg INIT FAILED:")
    traceback.print_exc()
    results["glamm_entityseg"] = False

# ── RefCOCO variants ─────────────────────────────────────────────────────────
try:
    from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
    for dsname in ["refcoco", "refcoco+", "refcocog"]:
        key = f"glamm_{dsname.replace('+','plus')}_val"
        try:
            ds = ReferSegmDataset(
                dataset_dir=DATA_ROOT,
                global_image_encoder=VISION_TOWER,
                image_size=1024,
                num_classes_per_sample=1,
                refer_segm_data=dsname,
                split="val",
            )
            results[key] = _check(f"GLaMM {dsname} val", ds)
        except Exception:
            print(f"\nGLaMM {dsname} INIT FAILED:")
            traceback.print_exc()
            results[key] = False
except Exception:
    print("\nGLaMM RefCOCO import FAILED:")
    traceback.print_exc()

# ── Material datasets ────────────────────────────────────────────────────────
try:
    from dataset.material_datasets.SynmatDataset import SynmatDataset
    from dataset.material_datasets.RealmatDataset import RealmatDataset
    from dataset.material_datasets.SamaDataset import SamaDataset

    for cls, name, kwargs in [
        (SynmatDataset, "synmat", {}),
        (RealmatDataset, "realmat", {}),
        (SamaDataset, "sama", {}),
    ]:
        key = f"glamm_{name}"
        try:
            ds = cls(use_star=True, **kwargs)
            results[key] = _check(f"GLaMM {name}", ds)
        except Exception:
            print(f"\nGLaMM {name} INIT FAILED:")
            traceback.print_exc()
            results[key] = False
except Exception:
    print("\nGLaMM material import FAILED:")
    traceback.print_exc()

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
all_pass = True
for k, v in results.items():
    status = "PASS" if v else "FAIL"
    print(f"  {status}  {k}")
    if not v:
        all_pass = False
print()
sys.exit(0 if all_pass else 1)
