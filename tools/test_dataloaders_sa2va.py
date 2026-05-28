"""
Smoke-test for all Sa2VA dataloaders (no GPU required).
Tests the inner GLaMM dataset via the wrapper's _glamm attribute.

Run from repo root:
  PYTHONPATH=Sa2VA Sa2VA/.venv/bin/python tools/test_dataloaders_sa2va.py
"""
import os, sys, traceback
import torch

VISION_TOWER = "openai/clip-vit-large-patch14-336"
QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N = 3

_SA2VA_ROOT = os.path.join(os.path.dirname(__file__), "..", "Sa2VA")
_GLAMM_ROOT = os.path.join(os.path.dirname(__file__), "..", "GLaMM")
_REPO_ROOT  = os.path.join(os.path.dirname(__file__), "..")
for _p in (_SA2VA_ROOT, _GLAMM_ROOT, _REPO_ROOT):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.paths import data_root

DATA_ROOT = str(data_root())

from model.llava import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

from transformers import AutoTokenizer, Qwen2_5_VLProcessor
from xtuner.utils import PROMPT_TEMPLATE

_tok_cfg = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=QWEN_MODEL,
    trust_remote_code=True,
    padding_side="right",
)
_proc_cfg = dict(
    type=Qwen2_5_VLProcessor.from_pretrained,
    pretrained_model_name_or_path=QWEN_MODEL,
    trust_remote_code=True,
)
_BASE = dict(
    tokenizer=_tok_cfg,
    prompt_template=PROMPT_TEMPLATE.qwen_chat,
    preprocessor=_proc_cfg,
    arch_type="qwen",
    repeats=1.0,
)


def _check_inner(name, ds_wrapper, n=N):
    """Check the inner GLaMM dataset (no full Qwen forward pass)."""
    inner = ds_wrapper._glamm
    print(f"\n{'='*60}")
    print(f"  {name}  (len={len(inner)})")
    print(f"{'='*60}")
    ok = 0
    for i in range(min(n, len(inner))):
        try:
            item = inner[i]
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
    status = "PASS" if ok == min(n, len(inner)) else f"FAIL ({ok}/{min(n,len(inner))} ok)"
    print(f"  -> {status}")
    return ok == min(n, len(inner))


results = {}

# ── EntitySeg ────────────────────────────────────────────────────────────────
try:
    from projects.sa2va.datasets.glamm_entityseg import Sa2VAEntitySegDataset

    ds = Sa2VAEntitySegDataset(
        samples_json=os.path.join(DATA_ROOT, "entityseg/entityseg_insseg_val.json"),
        global_image_encoder=VISION_TOWER,
        ann_train_json=os.path.join(DATA_ROOT, "entityseg/annotations/entityseg_insseg_train_annotations.json"),
        ann_val_json=os.path.join(DATA_ROOT, "entityseg/annotations/entityseg_insseg_val_annotations.json"),
        image_roots=[
            os.path.join(DATA_ROOT, "entityseg/images/entity_01_11580/images_merge"),
            os.path.join(DATA_ROOT, "entityseg/images/entity_02_11598/images"),
            os.path.join(DATA_ROOT, "entityseg/images/entity_03_10049/images_03_10049"),
        ],
        **_BASE,
    )
    results["sa2va_entityseg"] = _check_inner("Sa2VA EntitySeg", ds)
except Exception:
    print("\nSa2VA EntitySeg INIT FAILED:")
    traceback.print_exc()
    results["sa2va_entityseg"] = False

# ── RefCOCO variants ──────────────────────────────────────────────────────────
try:
    from projects.sa2va.datasets.glamm_refcoco import Sa2VARefCOCODataset

    for dsname in ["refcoco", "refcoco+", "refcocog"]:
        key = f"sa2va_{dsname.replace('+','plus')}_val"
        try:
            ds = Sa2VARefCOCODataset(
                dataset_dir=DATA_ROOT,
                refer_segm_data=dsname,
                split="val",
                image_size=1024,
                global_image_encoder=VISION_TOWER,
                **_BASE,
            )
            results[key] = _check_inner(f"Sa2VA {dsname} val", ds)
        except Exception:
            print(f"\nSa2VA {dsname} INIT FAILED:")
            traceback.print_exc()
            results[key] = False
except Exception:
    print("\nSa2VA RefCOCO import FAILED:")
    traceback.print_exc()

# ── Material datasets ─────────────────────────────────────────────────────────
try:
    from projects.sa2va.datasets.glamm_material import Sa2VAMaterialDataset

    for source in ["synmat", "realmat", "sama"]:
        key = f"sa2va_{source}"
        try:
            ds = Sa2VAMaterialDataset(
                source=source,
                base_data_dir=os.path.join(DATA_ROOT, "maoam_data"),
                use_star=True,
                **_BASE,
            )
            results[key] = _check_inner(f"Sa2VA {source}", ds)
        except Exception:
            print(f"\nSa2VA {source} INIT FAILED:")
            traceback.print_exc()
            results[key] = False
except Exception:
    print("\nSa2VA material import FAILED:")
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
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
