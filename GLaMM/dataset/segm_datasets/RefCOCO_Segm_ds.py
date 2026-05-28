import os
import argparse
import sys
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataset.utils.grefer import G_REFER
from dataset.utils.refcoco_refer import REFER
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS
from model.llava import conversation as conversation_lib
from pycocotools import mask
from tools.glamm_eval_utils import DEFAULT_IMAGE_TOKEN
from transformers import CLIPImageProcessor
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

_GLAMM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_GLAMM_ROOT, ".."))
for _p in (_GLAMM_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.hm_utils import augment_data


_REFCOCO_ACK_ONLY = {
    "yep",
    "yeah",
    "ya",
    "yup",
    "ok",
    "okay",
    "k",
    "sure",
    "thanks",
    "thank",
    "thank you",
    "thankyou",
    "nope",
    "nah",
    "yes",
    "no",
}
_REFCOCO_WS = re.compile(r"\s+")
_REFCOCO_EDGE_PUNCT = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")


def _normalize_refcoco_sentence(s: str) -> str:
    s = s.strip().lower()
    s = _REFCOCO_WS.sub(" ", s)
    s = _REFCOCO_EDGE_PUNCT.sub("", s)
    s = _REFCOCO_WS.sub(" ", s).strip()
    return s


def _is_bad_refcoco_sentence(s: str) -> bool:
    n = _normalize_refcoco_sentence(s)
    if n == "":
        return True
    if n in _REFCOCO_ACK_ONLY:
        return True
    return False


class ReferSegmDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        global_image_encoder,
        image_size: int = 1024,
        num_classes_per_sample: int = 3,
        refer_segm_data="refcoco||refcoco+||refcocog",
        split="val",
    ):
        self.dataset_dir = dataset_dir
        self.image_size = (int(image_size), int(image_size))
        self.num_classes_per_sample = int(num_classes_per_sample)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = (
            f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        )
        self.split = split
        self.initialize_refer_segm_data(refer_segm_data)
        self._build_index()

    def initialize_refer_segm_data(self, refer_segm_data):

        dataset_dir = os.path.join(self.dataset_dir, "Refer_Segm")
        self.refer_seg_ds_list = refer_segm_data.split("||")
        self.refer_segm_data = {}

        for dataset_name in self.refer_seg_ds_list:
            splitBy = "umd" if dataset_name == "refcocog" else "unc"
            refer_api = (
                G_REFER(dataset_dir, dataset_name, splitBy)
                if dataset_name == "grefcoco"
                else REFER(dataset_dir, dataset_name, splitBy)
            )
            ref_ids_train = refer_api.getRefIds(split=self.split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            refer_seg_ds = {
                "images": self.load_images(
                    refer_api,
                    images_ids_train,
                    dataset_dir,
                    dataset_name,
                ),
                "annotations": refer_api.Anns,
                "img2refs": self.create_img_to_refs_mapping(refs_train),
            }

            print(
                f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                f"images and {len(refer_seg_ds['annotations'])} annotations."
            )
            print(
                f'\033[92m----SEG-{self.split}:'
                f" Loaded ReferSeg - {dataset_name} dataset ----\033[0m"
            )

            self.refer_segm_data[dataset_name] = refer_seg_ds

    def load_images(
        self, refer_api, images_ids_train, dataset_dir, dataset_name
    ):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        for item in loaded_images:
            item = item.copy()
            if dataset_name == "refclef":
                item["file_name"] = os.path.join(
                    dataset_dir, "images", "saiapr_tc-12", item["file_name"]
                )
            else:
                item["file_name"] = os.path.join(
                    dataset_dir,
                    "images/mscoco/images/train2014",
                    item["file_name"],
                )
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [
                ref,
            ]
        return img2refs

    def __len__(self):
        return len(self._index)

    def _build_index(self):
        """Flatten (dataset_name, local_idx) into a single index for deterministic sampling."""
        index: list[tuple[str, int]] = []
        for dataset_name in self.refer_seg_ds_list:
            images = self.refer_segm_data[dataset_name]["images"]
            for local_idx in range(len(images)):
                index.append((dataset_name, local_idx))
        self._index = index

    def _extract_unique_refs(self, refs) -> list[tuple[int, list[str]]]:
        """
        Convert a list of REFER 'refs' for an image into a de-duplicated list of
        (ann_id, [sent1, sent2, ...]) after conservative ack-only filtering.
        """
        ref_candidates: list[tuple[int, list[str]]] = []
        for ref in refs:
            ann_id = ref["ann_id"]
            texts = [
                s["sent"]
                for s in ref.get("sentences", [])
                if s.get("sent") is not None and isinstance(s.get("sent"), str)
            ]
            seen_text = set()
            texts_dedup: list[str] = []
            for t in texts:
                t = t.strip()
                if not t:
                    continue
                if t in seen_text:
                    continue
                seen_text.add(t)
                texts_dedup.append(t)
            texts_dedup = [t for t in texts_dedup if not _is_bad_refcoco_sentence(t)]
            if len(texts_dedup) == 0:
                continue
            ref_candidates.append((ann_id, texts_dedup))

        seen_ann = set()
        ref_unique: list[tuple[int, list[str]]] = []
        for ann_id, texts in ref_candidates:
            key = str(ann_id)
            if key in seen_ann:
                continue
            seen_ann.add(key)
            ref_unique.append((ann_id, texts))

        return ref_unique

    def create_conversations(self, labels):
        questions = []
        answers = []
        for i, label in enumerate(labels):
            label = label.strip()
            assert len(label.split("||")) == 1
            questions.append(self.question_templates[0].format(class_name=label.lower()))
            answers.append(self.answer_list[0])

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations

    def __getitem__(self, idx):
        dataset_name, local_idx = self._index[idx % len(self._index)]

        refer_seg_ds = self.refer_segm_data[dataset_name]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        image_info = images[local_idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            raise ValueError(f"No refs for image_id={image_id} at idx={idx}")

        ref_unique = self._extract_unique_refs(refs)
        if len(ref_unique) < self.num_classes_per_sample:
            raise ValueError(f"Too few unique refs ({len(ref_unique)}) at idx={idx}")

        selected_refs = ref_unique[: self.num_classes_per_sample]

        sampled_sents: list[str] = []
        sampled_ann_ids: list[int] = []
        for ann_id, texts in selected_refs:
            sampled_sents.append(texts[0])
            sampled_ann_ids.append(ann_id)
        selected_labels = sampled_sents

        questions, conversations = self.create_conversations(selected_labels)

        masks_np = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks_np.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks_np.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks_np.append(m)

        masks = torch.from_numpy(np.stack(masks_np, axis=0)).float()  # [K,H,W]
        masks = (masks > 0).float()

        image_path = image_info["file_name"]
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float() / 255.0

        resize_shape = self.image_size[0]

        image = transforms.Resize(resize_shape)(image)
        masks = transforms.Resize(
            resize_shape, interpolation=InterpolationMode.NEAREST_EXACT
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

        if masks.sum() == 0:
            raise ValueError(f"Empty masks after augment at idx={idx}")

        grounding_image = image

        return {
            "filepath": image_path,
            "image_star": None,
            "image_without_star": grounding_image,  # [3,1024,1024] float
            "grounding_image": grounding_image,  # [3,1024,1024] float
            "global_enc_processor": self.global_enc_processor,
            "masks": masks,  # [K,1024,1024]
            "orig_size": self.image_size,
            "sampled_classes": selected_labels,
            "coords": [],
            "star": {"question": None, "conversation": None},
            "referring": {
                "question": questions,
                "conversation": conversations,  # list[str] length 1
                "desc": selected_labels,
            },
            "vqa": {"question": None, "answer": None, "conversation": None},
        }


def _save_rgb_tensor_as_png(rgb_chw: torch.Tensor, out_path: str) -> None:
    """Save float RGB CHW tensor in [0,1] as PNG."""
    if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] tensor, got {tuple(rgb_chw.shape)}")
    img = rgb_chw.detach().cpu().clamp(0, 1)
    img = (img * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()  # HWC RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(out_path, img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def _save_mask_tensor_as_png(mask_hw: torch.Tensor, out_path: str) -> None:
    """Save float/bool HW mask as PNG (0/255)."""
    if mask_hw.ndim != 2:
        raise ValueError(f"Expected [H,W] mask, got {tuple(mask_hw.shape)}")
    m = mask_hw.detach().cpu()
    if m.dtype != torch.uint8:
        m = (m > 0).to(torch.uint8) * 255
    ok = cv2.imwrite(out_path, m.numpy())
    if not ok:
        raise RuntimeError(f"Failed to write mask: {out_path}")


def main():
    """
    Debug helper:
    Loads N samples and saves the processed grounding image (+ first mask) to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root that contains Refer_Segm/")
    parser.add_argument("--out_dir", type=str, default="./refcoco_debug_out")
    parser.add_argument("--refer_segm_data", type=str, default="refcoco")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--global_image_encoder", type=str, default="openai/clip-vit-large-patch14-336")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = ReferSegmDataset(
        dataset_dir=args.dataset_dir,
        global_image_encoder=args.global_image_encoder,
        image_size=args.image_size,
        refer_segm_data=args.refer_segm_data,
        split=args.split,
    )

    n = min(args.num_samples, len(ds))
    for i in range(n):
        item = ds[i]
        img = item["grounding_image"]  # [3,1024,1024] float
        masks = item["masks"]          # [K,1024,1024]
        labels = item["sampled_classes"]

        base = os.path.join(args.out_dir, f"{i:03d}")
        _save_rgb_tensor_as_png(img, base + "_grounding.png")
        if masks is not None and masks.numel() > 0:
            _save_mask_tensor_as_png(masks[0], base + "_mask0.png")
        with open(base + "_labels.txt", "w") as f:
            for s in labels:
                f.write(str(s) + "\n")
        with open(base + "_path.txt", "w") as f:
            f.write(item["filepath"] + "\n")

    print(f"Saved {n} samples to: {args.out_dir}")


if __name__ == "__main__":
    GLAMM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if GLAMM_ROOT not in sys.path:
        sys.path.insert(0, GLAMM_ROOT)
    main()
