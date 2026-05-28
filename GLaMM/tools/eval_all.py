"""
GLaMM evaluation: RefCOCO / RefCOCO+ / RefCOCOg val+test splits, EntitySeg val.

Run from repo root:
  PYTHONPATH=. GLaMM/.venv/bin/torchrun --nproc_per_node=4 \\
    GLaMM/tools/eval_all.py \\
    --model_path MBZUAI/GLaMM-GranD-Pretrained \\
    --resume weights/glamm/mp_rank_00_model_states.pt \\
    --vision_pretrained weights/sam_vit_h_4b8939.pth

DATA_ROOT (default <repo>/data) must contain:
  entityseg/entityseg_insseg_val.json
  Refer_Segm/{refcoco,refcoco+,refcocog}/
  Refer_Segm/images/mscoco/images/train2014/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import partial
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import transformers
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import CenterCrop

_GLAMM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_GLAMM_ROOT, ".."))
# Insert repo root at position 0 so `utils/` package takes priority over
# any `utils.py` that torchrun may shadow it with (GLaMM/tools/utils.py).
for _p in (_REPO_ROOT, _GLAMM_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
    else:
        sys.path.remove(_p)
        sys.path.insert(0, _p)

from check_load import load_mp_rank_checkpoint
from dataset.datasets import custom_collate_fn_multi
from dataset.entity_datasets.Entity_InsSeg import EntitySegDataset
from dataset.material_datasets.SynmatDataset import SynmatDataset
from dataset.material_datasets.RealmatDataset import RealmatDataset
from dataset.material_datasets.SamaDataset import SamaDataset
from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from tools.glamm_eval_utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    Summary,
    intersectionAndUnionGPU,
)
from utils.paths import data_root

VISION_TOWER = "openai/clip-vit-large-patch14-336"
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _ddp_init():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def _is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Model loading (mirrors GLaMMDemo._setup_tokenizer + _setup_model)
# ---------------------------------------------------------------------------

def _load_tokenizer(model_path: str, model_max_length: int = 1536):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<bbox>", "[SEG]", "<p>", "</p>"]})
    return tokenizer


def _load_model(args, tokenizer, device):
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    model_args = dict(
        train_mask_decoder=True,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        seg_token_idx=seg_token_idx,
        vision_pretrained=args.vision_pretrained,
        vision_tower=VISION_TOWER,
        use_mm_start_end=True,
        mm_vision_select_layer=-2,
        pretrain_mm_mlp_adapter="",
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        mm_use_im_start_end=True,
        with_region=True,
        bbox_token_idx=bbox_token_idx,
        eop_token_idx=eop_token_idx,
        bop_token_idx=bop_token_idx,
        num_level_reg_features=4,
    )

    model = GLaMMForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load checkpoint BEFORE initialize_vision_modules so meta tensors (text_hidden_fcs,
    # grounding_encoder) are replaced at the checkpoint's dtype rather than re-initialized
    # as float32 by a second initialize_glamm_model call.
    _, state_dict = load_mp_rank_checkpoint(args.resume)
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] Checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=device)
    # Do NOT call initialize_glamm_model again — it was called in __init__ during
    # from_pretrained, and calling it a second time would overwrite text_hidden_fcs
    # and grounding_encoder with new float32 modules, breaking mixed-dtype inference.

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing: grounding encoder normalisation (applied after collation)
# ---------------------------------------------------------------------------

def _preprocess_batch(batch, device):
    def _move(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _move(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_move(x) for x in obj]
        return obj

    batch = _move(batch)

    if "grounding_enc_images" in batch and batch["grounding_enc_images"] is not None:
        g = (batch["grounding_enc_images"] * 255.0).to(torch.uint8).float()
        mean = IMG_MEAN.to(device)
        std = IMG_STD.to(device)
        g = (g - mean) / std
        h, w = g.shape[-2:]
        g = F.pad(g, (0, 1024 - w, 0, 1024 - h))
        batch["grounding_enc_images"] = g

    for key in ("global_enc_images", "images_star", "images_without_star"):
        t = batch.get(key)
        if isinstance(t, torch.Tensor) and t.dtype != torch.bfloat16:
            batch[key] = t.to(dtype=torch.bfloat16)

    if isinstance(batch.get("masks_list"), list):
        batch["masks_list"] = [
            m.to(dtype=torch.bfloat16) if isinstance(m, torch.Tensor) else m
            for m in batch["masks_list"]
        ]

    return batch


# ---------------------------------------------------------------------------
# Eval loop (one split, one task_key)
# ---------------------------------------------------------------------------

def _eval_split(
    model,
    dataset,
    task_key: str,
    tokenizer,
    batch_size: int,
    num_workers: int,
    device,
    rank: int,
    world_size: int,
    max_samples: int | None,
) -> dict:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=partial(
            custom_collate_fn_multi,
            tokenizer=tokenizer,
            use_mm_start_end=True,
            inference=True,
            local_rank=rank,
        ),
    )

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    giou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    per_rank_limit = -(-max_samples // world_size) if max_samples is not None else None
    n_processed = 0
    for batch in tqdm.tqdm(loader, disable=(rank != 0), total=per_rank_limit):
        if per_rank_limit is not None and n_processed >= per_rank_limit:
            break

        batch = _preprocess_batch(batch, device)

        if batch.get(task_key) is None:
            continue

        with torch.no_grad():
            results = model(**batch)

        task_out = results.get(task_key)
        if task_out is None or task_out.get("pred_masks") is None:
            continue

        pred_masks = task_out["pred_masks"]
        gt_masks_list = batch["masks_list"]

        for i, (gt_item, pred) in enumerate(zip(gt_masks_list, pred_masks)):
            if gt_item is None or pred is None:
                continue
            gt_masks = gt_item.int()           # [K, H, W]
            pred_mask = (pred > 0).int()       # [H, W] or [1, H, W]
            if pred_mask.ndim == 2:
                pred_mask = pred_mask.unsqueeze(0)  # [1, H, W]

            # Resize pred to gt size if needed
            if pred_mask.shape[-2:] != gt_masks.shape[-2:]:
                pred_mask = F.interpolate(
                    pred_mask.float().unsqueeze(0),
                    size=gt_masks.shape[-2:],
                    mode="nearest",
                ).squeeze(0).int()

            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for tgt, prd in zip(gt_masks, pred_mask):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prd.contiguous(), tgt.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                accuracy_iou[union_ == 0] += 1.0

            n_masks = gt_masks.shape[0]
            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            giou_meter.update((accuracy_iou / n_masks).cpu().numpy(), n=n_masks)
            n_processed += 1

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    giou_meter.all_reduce()

    iou_per_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = float(iou_per_class[1])
    giou = float(giou_meter.avg[1])
    return {"gIoU": giou, "cIoU": ciou}


# ---------------------------------------------------------------------------
# VQA eval loop (text generation, returns accuracy)
# ---------------------------------------------------------------------------

def _post_process_vqa(text: str) -> str:
    text = re.sub(r"[\n\r]+", " ", text).strip().upper()
    m = re.search(r"[A-D]", text)
    return m.group(0) if m else (text[0] if text else "")


def _normalize_gt_answer(answer: str) -> str:
    m = re.search(r"[A-D]", answer.upper())
    return m.group(0) if m else ""


def _eval_vqa_split(
    model,
    dataset,
    tokenizer,
    batch_size: int,
    num_workers: int,
    device,
    rank: int,
    world_size: int,
    max_samples: int | None,
) -> dict:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=partial(
            custom_collate_fn_multi,
            tokenizer=tokenizer,
            use_mm_start_end=True,
            inference=True,
            local_rank=rank,
        ),
    )

    per_rank_limit = -(-max_samples // world_size) if max_samples is not None else None
    n_correct = 0
    n_total = 0

    for batch in tqdm.tqdm(loader, disable=(rank != 0), total=per_rank_limit):
        if per_rank_limit is not None and n_total >= per_rank_limit:
            break

        batch = _preprocess_batch(batch, device)
        vqa_data = batch.get("vqa")
        if vqa_data is None or vqa_data.get("input_ids") is None:
            continue

        input_ids = vqa_data["input_ids"]   # (B, seq_len)
        labels = vqa_data["labels"]          # (B, seq_len)
        gt_answers = vqa_data["answers"]     # list of GT answer strings

        for i in range(input_ids.shape[0]):
            ids_1d = input_ids[i]
            lab_1d = labels[i]
            gt_raw = gt_answers[i] if i < len(gt_answers) else ""
            gt_letter = _normalize_gt_answer(gt_raw)
            if not gt_letter:
                continue

            # Find prompt boundary: first token where label != IGNORE_INDEX
            answer_positions = (lab_1d != -100).nonzero(as_tuple=False)
            if answer_positions.numel() == 0:
                continue
            prompt_len = int(answer_positions[0].item())
            prompt_ids = ids_1d[:prompt_len].unsqueeze(0).to(device)

            with torch.no_grad():
                generated_ids, _ = model.evaluate(
                    global_enc_images=batch["images_star"],
                    grounding_enc_images=batch["grounding_enc_images"],
                    input_ids=prompt_ids,
                    resize_list=batch.get("resize_list"),
                    orig_size=batch.get("orig_size"),
                    max_tokens_new=10,
                )

            new_ids = generated_ids[0, prompt_len:]
            pred_text = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
            pred_letter = _post_process_vqa(pred_text)

            if pred_letter:
                n_correct += int(pred_letter == gt_letter)
                n_total += 1

    # Gather across ranks
    counts = torch.tensor([n_correct, n_total], dtype=torch.int64, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    total_correct = int(counts[0].item())
    total_samples = int(counts[1].item())
    acc = float(total_correct) / float(total_samples) if total_samples > 0 else 0.0
    return {"accuracy": acc, "correct": total_correct, "total": total_samples}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="MBZUAI/GLaMM-GranD-Pretrained")
    p.add_argument("--resume", required=True, help="mp_rank_00_model_states.pt")
    p.add_argument("--vision_pretrained", required=True, help="Path to sam_vit_h_4b8939.pth")
    p.add_argument("--data_root", default=str(data_root()))
    p.add_argument("--out_dir", default=None, help="Output dir for summary JSONs (defaults to dirname(resume))")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--datasets", nargs="+",
                   choices=["entityseg", "refcoco", "refcocoplus", "refcocog", "synmat", "realmat", "sama"],
                   default=None,
                   help="Datasets to evaluate (default: all). refcoco=refcoco val/testA/testB, "
                        "refcocoplus=refcoco+ val/testA/testB, refcocog=refcocog val/test.")
    p.add_argument("--mat_tasks", nargs="+",
                   choices=["star", "referring", "vqa"],
                   default=["star", "referring", "vqa"],
                   help="Which tasks to evaluate on material datasets (default: all three).")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank, rank, world_size = _ddp_init()
    device = torch.device(f"cuda:{local_rank}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.resume))
    os.makedirs(out_dir, exist_ok=True)

    _ALL_DATASETS = {"entityseg", "refcoco", "refcocoplus", "refcocog", "synmat", "realmat", "sama"}
    run_datasets = set(args.datasets) if args.datasets else _ALL_DATASETS

    mat_tasks = set(args.mat_tasks)

    if _is_main(rank):
        print(f"[INFO] data_root={args.data_root}  out_dir={out_dir}  datasets={run_datasets}  mat_tasks={mat_tasks}")

    tokenizer = _load_tokenizer(args.model_path)
    model = _load_model(args, tokenizer, device)

    # (tag, dataset, task_key, eval_type) where eval_type is "mask" or "vqa_text"
    eval_targets: List[Tuple[str, object, str, str]] = []

    if "entityseg" in run_datasets:
        samples_json = os.path.join(args.data_root, "entityseg/entityseg_insseg_val.json")
        ann_val = os.path.join(args.data_root, "entityseg/annotations/entityseg_insseg_val_annotations.json")
        ann_train = os.path.join(args.data_root, "entityseg/annotations/entityseg_insseg_train_annotations.json")
        image_roots = [
            os.path.join(args.data_root, "entityseg/images/entity_01_11580/images_merge"),
            os.path.join(args.data_root, "entityseg/images/entity_02_11598/images"),
            os.path.join(args.data_root, "entityseg/images/entity_03_10049/images_03_10049"),
        ]
        ds = EntitySegDataset(
            samples_json=samples_json,
            use_star=True,
            use_referring=False,
            use_vqa=False,
            global_image_encoder=VISION_TOWER,
            ann_train_json=ann_train,
            ann_val_json=ann_val,
            image_roots=image_roots,
        )
        eval_targets.append(("entityseg_val", ds, "star", "mask"))

    # RefCOCO variants — one flag per variant
    _ref_flag_to_spec = {
        "refcoco":    ("refcoco",  ["val", "testA", "testB"]),
        "refcocoplus": ("refcoco+", ["val", "testA", "testB"]),
        "refcocog":   ("refcocog", ["val", "test"]),
    }
    for flag, (dsname, splits) in _ref_flag_to_spec.items():
        if flag not in run_datasets:
            continue
        for split in splits:
            tag = f"{dsname}_{split}".replace("+", "plus")
            ds = ReferSegmDataset(
                dataset_dir=args.data_root,
                global_image_encoder=VISION_TOWER,
                image_size=1024,
                num_classes_per_sample=1,
                refer_segm_data=dsname,
                split=split,
            )
            eval_targets.append((f"refcoco_{tag}", ds, "referring", "mask"))

    # Material variants — one flag per source, one dataset per source shared across tasks
    maoam_dir = os.path.join(args.data_root, "maoam_data")
    _mat_flag_to_cls = [
        ("synmat", SynmatDataset),
        ("realmat", RealmatDataset),
        ("sama", SamaDataset),
    ]
    for flag, cls in _mat_flag_to_cls:
        if flag not in run_datasets:
            continue
        ds = cls(
            base_data_dir=maoam_dir,
            use_star=("star" in mat_tasks or "vqa" in mat_tasks),
            use_referring=("referring" in mat_tasks),
            use_vqa=("vqa" in mat_tasks),
        )
        if "star" in mat_tasks:
            eval_targets.append((f"material_{flag}_star", ds, "star", "mask"))
        if "referring" in mat_tasks:
            eval_targets.append((f"material_{flag}_referring", ds, "referring", "mask"))
        if "vqa" in mat_tasks:
            eval_targets.append((f"material_{flag}_vqa", ds, "vqa", "vqa_text"))

    all_results = {}
    for tag, dataset, task_key, eval_type in eval_targets:
        if _is_main(rank):
            print(f"\n[INFO] Evaluating {tag} (task={task_key}, type={eval_type}, n={len(dataset)}) ...")
        dist.barrier()

        if eval_type == "vqa_text":
            metrics = _eval_vqa_split(
                model=model,
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                rank=rank,
                world_size=world_size,
                max_samples=args.max_samples,
            )
            if _is_main(rank):
                print(f"[RESULT] {tag}: accuracy={metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total']})")
        else:
            metrics = _eval_split(
                model=model,
                dataset=dataset,
                task_key=task_key,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                rank=rank,
                world_size=world_size,
                max_samples=args.max_samples,
            )
            if _is_main(rank):
                print(f"[RESULT] {tag}: gIoU={metrics['gIoU']:.4f}  cIoU={metrics['cIoU']:.4f}")

        all_results[tag] = metrics

        if _is_main(rank):
            out_path = os.path.join(out_dir, f"{tag}_summary.json")
            with open(out_path, "w") as f:
                json.dump({"tag": tag, **metrics}, f, indent=2)

    if _is_main(rank):
        combined_path = os.path.join(out_dir, "glamm_eval_summary.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[INFO] All results written to {combined_path}")
        for tag, m in all_results.items():
            if "accuracy" in m:
                print(f"  {tag:40s}  accuracy={m['accuracy']:.4f}")
            else:
                print(f"  {tag:40s}  gIoU={m['gIoU']:.4f}  cIoU={m['cIoU']:.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
