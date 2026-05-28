"""
Run *all* Sa2VA evaluations (RefCOCO-family val+test splits, EntitySeg val).

This is a thin orchestrator around the same "train-parity" model loading path used by:
  `projects/sa2va/gradio/app.py`

By default, you can provide only `--resume` and it will use a common default cfg.
If your checkpoint was trained with a different config, pass `--cfg`.

Single GPU:
  PYTHONPATH=. python projects/sa2va/evaluation/sa2va_eval_all.py \
    --resume weights/sa2va/mp_rank_00_model_states.pt

8 GPUs:
  PYTHONPATH=. torchrun --nproc_per_node=8 projects/sa2va/evaluation/sa2va_eval_all.py \
    --resume weights/sa2va/mp_rank_00_model_states.pt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX

from projects.sa2va.datasets.data_utils import sa2va_collect_fn_multitask
from projects.sa2va.datasets.glamm_refcoco import Sa2VARefCOCODataset
from projects.sa2va.datasets.glamm_entityseg import Sa2VAEntitySegDataset
from projects.sa2va.datasets.glamm_material import Sa2VAMaterialDataset
from utils.paths import data_root


_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_SA2VA_ROOT = os.path.abspath(os.path.join(_EVAL_DIR, "..", "..", ".."))
DEFAULT_CFG = os.path.join(_SA2VA_ROOT, "projects/sa2va/configs/glamm_qwen25_7b_material_only.py")


def _torch_load_ckpt_safely(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, TypeError) as e:
        print(
            f"Warning: torch.load(weights_only=True) failed for {path}: {e}\n"
            "Retrying with weights_only=False. Only do this if the checkpoint is from a trusted source."
        )
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _load_state_dict_from_mp_rank(path: str) -> dict:
    ckpt = _torch_load_ckpt_safely(path)
    if isinstance(ckpt, dict):
        if "module" in ckpt and isinstance(ckpt["module"], dict):
            sd = ckpt["module"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}: {type(sd)}")
    if len(sd) > 0 and all(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def _ddp_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ddp_init_if_needed():
    if _ddp_is_active():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _ddp_info() -> tuple[int, int, int]:
    if not _ddp_is_active():
        return (0, 1, 0)
    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return (rank, world, local_rank)


def _ddp_all_gather_object(obj):
    if not _ddp_is_active():
        return [obj]
    world = dist.get_world_size()
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, obj)
    return gathered


def _ddp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not _ddp_is_active():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _resolve_out_dir(resume_path: str, out_dir: Optional[str]) -> str:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    p = str(resume_path)
    if os.path.isdir(p):
        os.makedirs(p, exist_ok=True)
        return p
    d = os.path.dirname(p) if p else "."
    os.makedirs(d, exist_ok=True)
    return d


def _sanitize_filename(name: str) -> str:
    name = str(name).replace(os.sep, "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _post_process_vqa_answer(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[\n\r]+", " ", str(text)).strip()
    cleaned = cleaned.replace(".", " ").replace(",", " ")
    cleaned = cleaned.replace("option", " ")
    cleaned = re.sub(r"[^A-Za-z]", " ", cleaned)
    cleaned = cleaned.strip().upper()
    if not cleaned:
        return ""
    m = re.search(r"[A-D]", cleaned)
    return m.group(0) if m else cleaned[0]


def _normalize_vqa_answer(answer: str) -> str:
    if not answer:
        return ""
    cleaned = str(answer).replace(".", " ").strip().upper()
    m = re.search(r"[A-D]", cleaned)
    return m.group(0) if m else (cleaned[0] if cleaned else "")


def _mask_metrics_from_logits(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]

    pred_bin = (pred_mask > 0).to(torch.int64)
    gt_bin = (gt_mask > 0).to(torch.int64)

    valid = gt_mask.ne(255)
    valid_pixels = int(valid.sum().item())
    if valid_pixels > 0:
        pm_v = pred_bin[valid]
        gm_v = gt_bin[valid]
    else:
        pm_v = torch.zeros(0, dtype=torch.int64, device=pred_bin.device)
        gm_v = torch.zeros(0, dtype=torch.int64, device=pred_bin.device)

    tp = int(((pm_v == 1) & (gm_v == 1)).sum().item())
    fp = int(((pm_v == 1) & (gm_v == 0)).sum().item())
    fn = int(((pm_v == 0) & (gm_v == 1)).sum().item())
    inter = tp
    union = tp + fp + fn
    iou = (float(tp) / float(union)) if union > 0 else 1.0

    prec_den = tp + fp
    rec_den = tp + fn
    precision = (float(tp) / float(prec_den)) if prec_den > 0 else (1.0 if rec_den == 0 else 0.0)
    recall = (float(tp) / float(rec_den)) if rec_den > 0 else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (iou, precision, recall, f1, tp, fp, fn, inter, union)


def _prompt_len_from_labels(labels_1d: torch.Tensor) -> int:
    idx = (labels_1d != IGNORE_INDEX).nonzero(as_tuple=False)
    if idx.numel() == 0:
        raise RuntimeError("No supervised (answer) tokens found in labels; cannot run VQA generation.")
    return int(idx[0].item())


@torch.no_grad()
def _generate_vqa(
    *,
    sa2va_model,
    input_ids_1d: torch.Tensor,
    labels_1d: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, str, str]:
    device = next(sa2va_model.parameters()).device
    if not hasattr(sa2va_model, "mllm") or getattr(sa2va_model.mllm, "tokenizer", None) is None:
        raise RuntimeError("sa2va_model.mllm.tokenizer is not initialized; cannot decode VQA.")
    tok = sa2va_model.mllm.tokenizer

    prompt_len = _prompt_len_from_labels(labels_1d)
    prompt_ids = input_ids_1d[:prompt_len].unsqueeze(0).to(device=device)
    attn = torch.ones_like(prompt_ids, dtype=torch.long, device=device)

    gt_ids = input_ids_1d[prompt_len:]
    if gt_ids.numel() == 0:
        raise RuntimeError("No GT answer tokens found after prompt_len; cannot compute VQA accuracy.")
    if int((gt_ids == 0).any().item()):
        first_pad = int((gt_ids == 0).nonzero(as_tuple=False)[0].item())
        gt_ids = gt_ids[:first_pad]
    gt_text = tok.decode(gt_ids.tolist(), skip_special_tokens=True).strip()
    gt_letter = _normalize_vqa_answer(gt_text)
    if gt_letter == "":
        raise RuntimeError(f"Failed to parse GT VQA answer letter from gt_text={gt_text!r}")

    pv = pixel_values.to(device=device, dtype=torch.bfloat16)
    g = image_grid_thw.to(device=device, dtype=torch.long)

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen = sa2va_model.mllm.model.generate(
                input_ids=prompt_ids,
                attention_mask=attn,
                pixel_values=pv,
                image_grid_thw=g,
                do_sample=False,
                num_beams=1,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
            )
    else:
        gen = sa2va_model.mllm.model.generate(
            input_ids=prompt_ids,
            attention_mask=attn,
            pixel_values=pv,
            image_grid_thw=g,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
        )
    new_tokens = gen[0, prompt_ids.shape[1] :]
    if new_tokens.numel() == 0:
        raise RuntimeError("Model.generate produced 0 new tokens (empty answer).")
    pred_text = tok.decode(new_tokens.tolist(), skip_special_tokens=True).strip()
    if pred_text == "":
        raise RuntimeError("Decoded VQA answer text is empty.")
    pred_letter = _post_process_vqa_answer(pred_text)
    if pred_letter == "":
        raise RuntimeError(f"Failed to parse predicted VQA answer letter from pred_text={pred_text!r}")
    return (pred_letter, pred_text, gt_letter)


@torch.no_grad()
def eval_one_loader(
    *,
    sa2va_model,
    loader: DataLoader,
    eval_name: str,
    out_dir: str,
    max_new_tokens: int,
    disable_vqa: bool,
) -> dict:
    rank, world, _ = _ddp_info()

    per_sample: List[dict] = []
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    vqa_tot = 0
    vqa_cor = 0
    vqa_by_key: Dict[str, Dict[str, int]] = {}

    it = loader
    if rank == 0:
        try:
            import tqdm  # type: ignore

            it = tqdm.tqdm(loader, desc=eval_name, total=len(loader))
        except Exception:
            it = loader

    for batch_wrap in it:
        batch = batch_wrap["data"]
        batch["inference"] = True
        out = sa2va_model(batch, None, mode="loss")

        base_meta = batch.get("meta", None) or [None for _ in range(len(batch.get("src", []) or []))]
        base_src = batch.get("src", None) or ["unknown" for _ in range(len(base_meta))]

        for task_name in ["star", "referring"]:
            if task_name not in out:
                continue
            task_out = out.get(task_name) or {}
            pred_masks = task_out.get("pred_masks", None)
            gt_masks = task_out.get("gt_masks", None)
            base_indices = (batch.get("tasks", {}).get(task_name, {}) or {}).get("base_indices", None)
            if pred_masks is None or gt_masks is None or base_indices is None:
                continue
            if len(pred_masks) != len(gt_masks):
                raise RuntimeError(f"{eval_name}: pred/gt len mismatch for task={task_name}")

            base_indices_list = base_indices.detach().cpu().tolist()
            for i, (pm, gm) in enumerate(zip(pred_masks, gt_masks)):
                if pm is None or gm is None:
                    continue
                iou, precision, recall, f1, tp, fp, fn, inter, union = _mask_metrics_from_logits(pm, gm)
                bidx = int(base_indices_list[i])
                src = str(base_src[bidx]) if bidx < len(base_src) else "unknown"
                meta = base_meta[bidx] if bidx < len(base_meta) else None

                agg.setdefault(src, {}).setdefault(task_name, {})
                a = agg[src][task_name]
                a["tp"] = a.get("tp", 0.0) + tp
                a["fp"] = a.get("fp", 0.0) + fp
                a["fn"] = a.get("fn", 0.0) + fn
                a["inter"] = a.get("inter", 0.0) + inter
                a["union"] = a.get("union", 0.0) + union
                a["n"] = a.get("n", 0.0) + 1.0
                a["sum_iou"] = a.get("sum_iou", 0.0) + float(iou)

                per_sample.append(
                    {
                        "eval": eval_name,
                        "dataset": src,
                        "task": task_name,
                        "iou": float(iou),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "inter": int(inter),
                        "union": int(union),
                        "meta": meta,
                    }
                )

        if (not disable_vqa) and ("vqa" in batch.get("tasks", {})):
            vqa_task = batch["tasks"]["vqa"]
            input_ids = vqa_task["input_ids"]
            labels = vqa_task["labels"]
            base_indices = vqa_task["base_indices"]
            pv_list = vqa_task["pixel_values"]
            g_list = vqa_task["image_grid_thw"]

            base_indices_list = base_indices.detach().cpu().tolist()
            for i in range(int(input_ids.shape[0])):
                ids_1d = input_ids[i].detach().cpu()
                lab_1d = labels[i].detach().cpu()
                bidx = int(base_indices_list[i])
                src = str(base_src[bidx]) if bidx < len(base_src) else "unknown"
                meta = base_meta[bidx] if bidx < len(base_meta) else None

                pred_letter, pred_text, gt_letter = _generate_vqa(
                    sa2va_model=sa2va_model,
                    input_ids_1d=ids_1d,
                    labels_1d=lab_1d,
                    pixel_values=pv_list[i],
                    image_grid_thw=g_list[i],
                    max_new_tokens=max_new_tokens,
                )
                correct = (pred_letter == gt_letter)
                vqa_tot += 1
                vqa_cor += int(correct)

                q_idx = None
                if isinstance(meta, dict) and meta.get("q_idx", None) is not None:
                    q_idx = meta.get("q_idx")
                q_tag = "q1" if q_idx == 0 else ("q2" if q_idx == 1 else "q?")
                key = f"{src}/{q_tag}"
                vqa_by_key.setdefault(key, {"tot": 0, "cor": 0})
                vqa_by_key[key]["tot"] += 1
                vqa_by_key[key]["cor"] += int(correct)

                per_sample.append(
                    {
                        "eval": eval_name,
                        "dataset": src,
                        "task": "vqa",
                        "pred": pred_letter,
                        "gt": gt_letter,
                        "correct": bool(correct),
                        "pred_text": pred_text,
                        "meta": meta,
                    }
                )

    gathered = _ddp_all_gather_object(per_sample)
    if rank == 0:
        per_sample_all = [r for part in gathered for r in (part or [])]
    else:
        per_sample_all = []

    agg_gathered = _ddp_all_gather_object(agg)
    vqa_vec = torch.tensor([vqa_tot, vqa_cor], dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
    vqa_vec = _ddp_all_reduce_sum(vqa_vec)
    vqa_tot_all = int(vqa_vec[0].item())
    vqa_cor_all = int(vqa_vec[1].item())
    vqa_acc = (float(vqa_cor_all) / float(vqa_tot_all)) if vqa_tot_all > 0 else 0.0

    vqa_by_key_all: Dict[str, Dict[str, int | float]] = {}
    vqa_by_key_gathered = _ddp_all_gather_object(vqa_by_key)

    if rank == 0:
        agg_merged: Dict[str, Dict[str, Dict[str, float]]] = {}
        for part in agg_gathered:
            if not isinstance(part, dict):
                continue
            for ds, tasks in part.items():
                agg_merged.setdefault(ds, {})
                for tn, sums in tasks.items():
                    agg_merged[ds].setdefault(tn, {})
                    for k, v in sums.items():
                        agg_merged[ds][tn][k] = agg_merged[ds][tn].get(k, 0.0) + float(v)

        agg_final: Dict[str, Dict[str, dict]] = {}
        for ds, tasks in agg_merged.items():
            agg_final.setdefault(ds, {})
            for tn, s in tasks.items():
                tp = int(s.get("tp", 0.0))
                fp = int(s.get("fp", 0.0))
                fn = int(s.get("fn", 0.0))
                inter = float(s.get("inter", 0.0))
                union = float(s.get("union", 0.0))
                n = float(s.get("n", 0.0))
                sum_iou = float(s.get("sum_iou", 0.0))
                ciou = (inter / union) if union > 0 else 0.0
                giou = (sum_iou / n) if n > 0 else 0.0
                prec_den = tp + fp
                rec_den = tp + fn
                precision = (float(tp) / float(prec_den)) if prec_den > 0 else (1.0 if rec_den == 0 else 0.0)
                recall = (float(tp) / float(rec_den)) if rec_den > 0 else 1.0
                f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                agg_final[ds][tn] = {
                    "n": int(n),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "ciou": float(ciou),
                    "giou": float(giou),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }

        merged_vk: Dict[str, Dict[str, int]] = {}
        for part in vqa_by_key_gathered:
            if not isinstance(part, dict):
                continue
            for k, v in part.items():
                merged_vk.setdefault(k, {"tot": 0, "cor": 0})
                merged_vk[k]["tot"] += int(v.get("tot", 0))
                merged_vk[k]["cor"] += int(v.get("cor", 0))
        for k, v in merged_vk.items():
            tot = int(v["tot"])
            cor = int(v["cor"])
            vqa_by_key_all[k] = {"tot": tot, "cor": cor, "acc": (float(cor) / float(tot)) if tot > 0 else 0.0}

        stem = _sanitize_filename(eval_name)
        jsonl_path = os.path.join(out_dir, f"{stem}.jsonl")
        summary_path = os.path.join(out_dir, f"{stem}_summary.json")

        with open(jsonl_path, "w") as f:
            for r in per_sample_all:
                f.write(json.dumps(r) + "\n")

        summary = {
            "eval": eval_name,
            "world_size": int(world),
            "segmentation": agg_final,
            "vqa": {
                "enabled": (not disable_vqa),
                "tot": int(vqa_tot_all),
                "cor": int(vqa_cor_all),
                "acc": float(vqa_acc),
                "by_key": vqa_by_key_all,
            },
            "outputs": {"per_sample_jsonl": jsonl_path, "summary_json": summary_path},
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    return {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, help="DeepSpeed mp_rank_00_model_states.pt (or a folder containing it).")
    p.add_argument("--cfg", default=DEFAULT_CFG, help=f"MMEngine config used for training (default: {DEFAULT_CFG}).")
    p.add_argument("--out_dir", default=None, help="Output directory (default: dirname(--resume)).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--disable_vqa", action="store_true")
    p.add_argument("--mat_tasks", nargs="+",
                   choices=["star", "referring", "vqa"],
                   default=["star", "referring", "vqa"],
                   help="Which tasks to evaluate on material datasets (default: all three).")
    p.add_argument("--max_samples", type=int, default=None, help="Cap total samples per dataset (for smoke tests).")
    p.add_argument("--datasets", nargs="+",
                   choices=["entityseg", "refcoco", "refcocoplus", "refcocog", "synmat", "realmat", "sama"],
                   default=None,
                   help="Datasets to evaluate (default: all). refcoco=refcoco val/testA/testB, "
                        "refcocoplus=refcoco+ val/testA/testB, refcocog=refcocog val/test.")
    p.add_argument("--work_dir", default=os.environ.get("WORK_DIR"),
                   help="Root for model artifacts (SAM2 ckpt, pretrained weights). Overrides config WORK_DIR.")
    p.add_argument("--data_root", default=os.environ.get("DATA_ROOT"),
                   help="Root for datasets. Overrides config DATA_DIR.")
    return p.parse_args()


def _build_loader(ds, *, batch_size: int, num_workers: int, max_samples: Optional[int] = None) -> DataLoader:
    from torch.utils.data import Subset
    rank, world, _ = _ddp_info()
    if max_samples is not None and len(ds) > max_samples:
        ds = Subset(ds, list(range(max_samples)))
    sampler = None
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        collate_fn=sa2va_collect_fn_multitask,
        drop_last=False,
    )


def main():
    _ddp_init_if_needed()
    rank, _, local_rank = _ddp_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    args = parse_args()
    out_dir = _resolve_out_dir(str(args.resume), args.out_dir)

    cfg = Config.fromfile(args.cfg)

    if args.work_dir:
        wdir = args.work_dir
        cfg.model.grounding_encoder.ckpt_path = os.path.join(wdir, "weights/sam2/sam2_hiera_large.pt")
        if cfg.model.get("pretrained_pth"):
            cfg.model.pretrained_pth = None

    sa2va_model = BUILDER.build(cfg.model)
    state_dict = _load_state_dict_from_mp_rank(str(args.resume))
    missing, unexpected = sa2va_model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"[INFO] Loaded checkpoint: {args.resume}")
        print(f"[INFO] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    sa2va_model = sa2va_model.eval().cuda()

    if not hasattr(cfg, "sa2va_glamm_default_dataset_configs"):
        raise ValueError("Config missing `sa2va_glamm_default_dataset_configs` (expected glamm_* configs).")
    base_ds_cfg = cfg.sa2va_glamm_default_dataset_configs

    data_root_path = os.path.expanduser(args.data_root) if args.data_root else str(data_root())
    _ALL_DATASETS = {"entityseg", "refcoco", "refcocoplus", "refcocog", "synmat", "realmat", "sama"}
    run_datasets = set(args.datasets) if args.datasets else _ALL_DATASETS
    mat_tasks = set(args.mat_tasks)

    eval_targets: List[Tuple[str, Any]] = []

    if "entityseg" in run_datasets:
        samples_json = os.path.join(data_root_path, "entityseg/entityseg_insseg_val.json")
        if args.data_root is None and hasattr(cfg, "entityseg_val_dataset") and isinstance(cfg.entityseg_val_dataset, dict):
            samples_json = str(cfg.entityseg_val_dataset.get("samples_json", samples_json))
        eval_targets.append(
            (
                "entityseg_val",
                Sa2VAEntitySegDataset(samples_json=samples_json, repeats=1.0, **base_ds_cfg),
            )
        )

    # RefCOCO variants — one flag per variant
    dataset_dir = data_root_path
    global_image_encoder = "openai/clip-vit-large-patch14-336"
    image_size = 1024
    if args.data_root is None and hasattr(cfg, "refcoco_val_dataset") and isinstance(cfg.refcoco_val_dataset, dict):
        dataset_dir = str(cfg.refcoco_val_dataset.get("dataset_dir", dataset_dir))
        global_image_encoder = str(cfg.refcoco_val_dataset.get("global_image_encoder", global_image_encoder))
        image_size = int(cfg.refcoco_val_dataset.get("image_size", image_size))

    _ref_flag_to_spec = {
        "refcoco":     ("refcoco",  ["val", "testA", "testB"]),
        "refcocoplus": ("refcoco+", ["val", "testA", "testB"]),
        "refcocog":    ("refcocog", ["val", "test"]),
    }
    for flag, (dsname, splits) in _ref_flag_to_spec.items():
        if flag not in run_datasets:
            continue
        for split in splits:
            tag = f"{dsname}_{split}".replace("+", "plus")
            eval_targets.append(
                (
                    f"refcoco_{tag}",
                    Sa2VARefCOCODataset(
                        dataset_dir=dataset_dir,
                        refer_segm_data=dsname,
                        split=split,
                        image_size=image_size,
                        global_image_encoder=global_image_encoder,
                        src_name=f"refcoco_{tag}",
                        repeats=1.0,
                        **base_ds_cfg,
                    ),
                )
            )

    # Material variants — one dataset per source shared across tasks
    maoam_dir = os.path.join(data_root_path, "maoam_data")
    _mat_desc_jsons = {
        "synmat": os.path.join(maoam_dir, "synmat_descriptions.json"),
        "realmat": os.path.join(maoam_dir, "realmat_descriptions.json"),
        "sama":    os.path.join(maoam_dir, "sama_descriptions.json"),
    }
    _mat_vqa_jsons = {
        "synmat": os.path.join(maoam_dir, "synmat_vqa.json"),
        "realmat": os.path.join(maoam_dir, "realmat_vqa.json"),
        "sama":    os.path.join(maoam_dir, "sama_vqa.json"),
    }
    for source in ["synmat", "realmat", "sama"]:
        if source not in run_datasets:
            continue
        eval_targets.append(
            (
                f"material_{source}",
                Sa2VAMaterialDataset(
                    source=source,
                    base_data_dir=maoam_dir,
                    use_star=("star" in mat_tasks or "vqa" in mat_tasks),
                    use_referring=("referring" in mat_tasks),
                    use_vqa=("vqa" in mat_tasks),
                    description_json=_mat_desc_jsons[source],
                    vqa_json=_mat_vqa_jsons[source],
                    repeats=1.0,
                    **base_ds_cfg,
                ),
            )
        )

    results_index = {
        "resume": str(args.resume),
        "cfg": str(args.cfg),
        "out_dir": out_dir,
        "runs": [],
    }

    for eval_name, ds in eval_targets:
        loader = _build_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers, max_samples=args.max_samples)
        summary = eval_one_loader(
            sa2va_model=sa2va_model,
            loader=loader,
            eval_name=eval_name,
            out_dir=out_dir,
            max_new_tokens=int(args.max_new_tokens),
            disable_vqa=bool(args.disable_vqa),
        )
        if rank == 0:
            results_index["runs"].append(summary.get("outputs", {}))

    if rank == 0:
        index_path = os.path.join(out_dir, "sa2va_eval_all_index.json")
        with open(index_path, "w") as f:
            json.dump(results_index, f, indent=2)
        print(f"[OK] wrote index: {index_path}")


if __name__ == "__main__":
    main()
