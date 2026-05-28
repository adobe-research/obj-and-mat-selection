"""
EntitySeg (EntityV2) instance segmentation loader for GLaMM.

This dataloader reads an *expanded* per-instance JSON (one row per mask), similar
to the material expanded format you showed (file, mask_id, area, h, w, ...).

Important constraints (as discussed):
- We no longer sample K masks per image; each dataset row is one instance (K=1).
- The expanded JSON is assumed to have already filtered:
  - remove==1
  - iscrowd==1
  - tiny masks (<0.1% of image area)
  - empty masks (area<=0)
- Prompt/task payloads are left as a skeleton (None) for now; we only implement
  image+mask loading and preprocessing.

Preprocessing (matches Evermotion):
- train: resize shortest side to random in [1024..1536], then random crop 1024x1024
- val  : resize shortest side to 1024, then center crop 1024x1024

Mask resolution always tracks image resolution (NEAREST_EXACT).
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPImageProcessor
from pycocotools import mask as mask_util

_GLAMM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_GLAMM_ROOT, ".."))
for _p in (_GLAMM_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.hm_utils import augment_data, add_star_marker
from dataset.utils.utils import ANSWER_LIST, ENTITY_SEG_QUESTIONS
from tools.glamm_eval_utils import DEFAULT_IMAGE_TOKEN
from model.llava import conversation as conversation_lib




DATA_ROOT = os.environ.get("DATA_ROOT", "data")

DEFAULT_TRAIN_ANN_JSON = f"{DATA_ROOT}/entityseg/annotations/entityseg_insseg_train_annotations.json"
DEFAULT_VAL_ANN_JSON = f"{DATA_ROOT}/entityseg/annotations/entityseg_insseg_val_annotations.json"

DEFAULT_IMAGE_ROOTS = [
    f"{DATA_ROOT}/entityseg/images/entity_01_11580/images_merge",
    f"{DATA_ROOT}/entityseg/images/entity_02_11598/images",
    f"{DATA_ROOT}/entityseg/images/entity_03_10049/images_03_10049",
]


class EntitySegDataset(torch.utils.data.Dataset):
    """
    Per-instance EntitySeg loader (one mask per sample).

    Reads expanded samples JSON (one row per mask_id). Mask decoding uses the
    *original* COCO annotation JSONs and a safe key of (file_name, mask_id),
    because annotation ids overlap between original train/val.
    """

    def __init__(
        self,
        samples_json: str,
        image_size: Tuple[int, int] = (1024, 1024),
        marker_size: int = 32,
        use_star: bool = True,
        use_referring: bool = False,
        use_vqa: bool = False,
        global_image_encoder: str = "openai/clip-vit-large-patch14-336",
        ann_train_json: str = DEFAULT_TRAIN_ANN_JSON,
        ann_val_json: str = DEFAULT_VAL_ANN_JSON,
        image_roots: Optional[list[str]] = None,
    ):
        super().__init__()
        self.samples_json = samples_json
        self.image_size = tuple(image_size)
        self.marker_size = int(marker_size)
        self.use_star = bool(use_star)
        self.use_referring = bool(use_referring)
        self.use_vqa = bool(use_vqa)

        self.global_image_encoder = global_image_encoder
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(
            self.global_image_encoder
        )

        self.image_roots = list(image_roots) if image_roots is not None else list(DEFAULT_IMAGE_ROOTS)

        data = json.load(open(self.samples_json, "r"))
        self.samples = data["samples"]

        self.ann_lookup: Dict[tuple[str, int], dict] = {}
        self._load_ann_lookup(ann_train_json)
        self._load_ann_lookup(ann_val_json)

        self.category_id_to_name: Dict[int, str] = {}
        self._load_categories(ann_train_json)
        if not self.category_id_to_name:
            self._load_categories(ann_val_json)
        if not self.category_id_to_name:
            raise ValueError(
                "Failed to load any categories from COCO annotation jsons. "
                f"Tried ann_train_json={ann_train_json} and ann_val_json={ann_val_json}."
            )

        self._img_path_cache: Dict[str, str] = {}

        print(f"[EntitySegDataset] Loaded {len(self.samples)} samples from {self.samples_json}")

    def _load_categories(self, ann_json_path: str) -> None:
        if not os.path.exists(ann_json_path):
            return
        d = json.load(open(ann_json_path, "r"))
        for cat in d.get("categories", []) or []:
            cid = int(cat["id"])
            name = str(cat["name"]).strip()
            if name:
                self.category_id_to_name[cid] = name

    def _create_conversation(self, question: str, answer: str) -> list[str]:
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        return [conv.get_prompt()]

    def _sample_star_coords(self, mask: torch.Tensor) -> list[list[int]]:
        """Deterministic: place 1 star at the valid candidate closest to the mask centroid."""
        if mask.dtype != torch.bool:
            mask = mask.bool()
        H, W = int(mask.shape[-2]), int(mask.shape[-1])
        if H == 0 or W == 0:
            return []

        invalid = (~mask).to(torch.float32).unsqueeze(0).unsqueeze(0)
        for k in (12, 6, 3, 1):
            if k > H or k > W:
                continue
            pooled = F.max_pool2d(invalid, kernel_size=k, stride=1, padding=0)[0, 0]
            ys, xs = torch.where(pooled == 0)
            if ys.numel() > 0:
                break
        if ys.numel() == 0:
            return []

        center_h = ys + (k // 2)
        center_w = xs + (k // 2)
        cent_h = center_h.float().mean().round().long()
        cent_w = center_w.float().mean().round().long()
        dists = (center_h - cent_h).abs() + (center_w - cent_w).abs()
        best = dists.argmin()
        return [[int(center_h[best].item()), int(center_w[best].item())]]

    def _load_ann_lookup(self, ann_json_path: str) -> None:
        if not os.path.exists(ann_json_path):
            return
        d = json.load(open(ann_json_path, "r"))
        img_id_to_file = {img["id"]: img["file_name"] for img in d.get("images", [])}
        for ann in d.get("annotations", []):
            img_id = ann.get("image_id")
            fn = img_id_to_file.get(img_id)
            if fn is None:
                continue
            ann_id = int(ann.get("id"))
            key = (fn, ann_id)
            if key not in self.ann_lookup:
                self.ann_lookup[key] = ann

    def _resolve_image_path(self, file_name: str) -> str:
        if file_name in self._img_path_cache:
            return self._img_path_cache[file_name]
        if os.path.isabs(file_name) and os.path.exists(file_name):
            self._img_path_cache[file_name] = file_name
            return file_name
        for root in self.image_roots:
            p = os.path.join(root, file_name)
            if os.path.exists(p):
                self._img_path_cache[file_name] = p
                return p
        raise FileNotFoundError(f"Could not locate image file: {file_name} (roots={self.image_roots})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for attempt in range(1):
            idx = index % len(self.samples)
            s = self.samples[idx]
            file_name = s["file"]
            mask_id = int(s["mask_id"])

            ann = self.ann_lookup.get((file_name, mask_id))
            if ann is None:
                raise KeyError(
                    f"Annotation not found for (file={file_name}, mask_id={mask_id})"
                )

            img_path = self._resolve_image_path(file_name)

            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image at {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                / 255.0
            )  # [3,H,W]

            seg = ann.get("segmentation")
            if not isinstance(seg, dict):
                raise ValueError(
                    f"Expected RLE dict segmentation for mask_id={mask_id}, got {type(seg)}"
                )

            m = mask_util.decode(seg)
            if m.ndim == 3:
                m = np.sum(m, axis=2)
            m = (m > 0).astype(np.uint8)
            masks = torch.from_numpy(m).unsqueeze(0).float()  # [1,H,W]

            image = transforms.Resize(self.image_size[0])(image)
            masks = transforms.Resize(
                self.image_size[0], interpolation=InterpolationMode.NEAREST_EXACT
            )(masks)

            image, masks = augment_data(
                image,
                masks,
                size=self.image_size,
                flip=False,
                test=True,
                crop=False,
                resize=False,
                point_hw=None,
            )

            mask_bool = masks[0] > 0.5
            coords = self._sample_star_coords(mask_bool)
            if len(coords) == 0:
                raise ValueError(f"No valid star candidates at index={index}, file={file_name}")

            break

        image_for_sam = image.clone()
        image_with_star = None
        first_color = None
        if self.use_star:
            image_with_star = image.clone()
            for (h, w) in coords:
                image_with_star, star_color = add_star_marker(
                    image_with_star, h, w, size=self.marker_size, color=first_color
                )
                if first_color is None:
                    first_color = star_color

        orig_size = (self.image_size[0], self.image_size[1])
        category_id = int(ann.get("category_id", -1))

        question = None
        conversation = None
        if self.use_star:
            class_name = self.category_id_to_name.get(category_id, str(category_id))
            begin_str = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"
            question = begin_str + ENTITY_SEG_QUESTIONS[0].format(class_name=class_name.lower()).replace(
                "<COLOR>", str(first_color)
            )
            answer = ANSWER_LIST[0]
            conversation = self._create_conversation(question, answer)

        return {
            "filepath": img_path,
            "image_star": image_with_star,        # CLIP path (with star) or None
            "image_without_star": image_for_sam,  # CLIP path (no star)
            "grounding_image": image_for_sam,   # SAM / grounding path (no star)
            "global_enc_processor": self.global_enc_processor,
            "masks": masks,                     # [1,1024,1024]
            "orig_size": orig_size,
            "sampled_classes": [category_id],
            "coords": coords,
            "star": {"question": question, "conversation": conversation}
            if self.use_star
            else {"question": None, "conversation": None},
            "referring": {"question": None, "conversation": None, "desc": None}
            if not self.use_referring
            else {"question": None, "conversation": None, "desc": None},
            "vqa": {"question": None, "answer": None, "conversation": None}
            if not self.use_vqa
            else {"question": None, "answer": None, "conversation": None},
        }


def main():
    """
    Debug helper:
    Loads N samples and saves:
    - grounding image (no star)  -> *_image.png
    - star image (with star)     -> *_image_star.png
    - first mask                 -> *_mask.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_json", type=str, required=True, help="Expanded per-instance samples JSON")
    parser.add_argument("--out_dir", type=str, default="./entityseg_debug_out")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--global_image_encoder", type=str, default="openai/clip-vit-large-patch14-336")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = EntitySegDataset(
        samples_json=args.samples_json,
        global_image_encoder=args.global_image_encoder,
    )

    n = min(args.num_samples, len(ds))
    for i in range(n):
        item = ds[i]
        img = item["grounding_image"]  # [3,1024,1024] float (no star)
        img_star = item.get("image_star")  # [3,1024,1024] float (with star)
        masks = item["masks"]          # [1,1024,1024]

        base = os.path.join(args.out_dir, f"{i:03d}")
        img_np = img.detach().cpu().clamp(0, 1)
        img_np = (img_np * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(base + "_image.png", img_bgr)

        if img_star is not None:
            img_star_np = img_star.detach().cpu().clamp(0, 1)
            img_star_np = (img_star_np * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
            img_star_bgr = cv2.cvtColor(img_star_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(base + "_image_star.png", img_star_bgr)

        if masks is not None and masks.numel() > 0:
            m0 = (masks[0].detach().cpu().numpy() > 0).astype(np.uint8) * 255
            cv2.imwrite(base + "_mask.png", m0)

        with open(base + "_meta.txt", "w") as f:
            f.write(f"filepath: {item['filepath']}\n")
            f.write(f"category_id: {item['sampled_classes']}\n")
            f.write(f"coords: {item.get('coords', [])}\n")
            f.write(f"star_question: {item.get('star', {}).get('question')}\n")
            conv = item.get("star", {}).get("conversation")
            conv0 = conv[0] if isinstance(conv, list) and len(conv) > 0 else conv
            f.write(f"star_conversation: {conv0}\n")

    print(f"Saved {n} samples to: {args.out_dir}")


if __name__ == "__main__":
    main()
