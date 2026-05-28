import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from .base import Sa2VABaseDataset


_GLAMM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "GLaMM"))
_REPO_ROOT = os.path.abspath(os.path.join(_GLAMM_ROOT, ".."))
for _p in (_GLAMM_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.entity_datasets.Entity_InsSeg import (  # noqa: E402
    DEFAULT_TRAIN_ANN_JSON,
    DEFAULT_VAL_ANN_JSON,
)


@dataclass
class _TaskPack:
    input_ids: list[int]
    labels: list[int]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    conv_prompt: str
    base_idx: int


def _extract_answer_from_llava_prompt(prompt: str) -> str:
    k = "ASSISTANT:"
    i = prompt.rfind(k)
    if i == -1:
        print("\n==== DEBUG: unexpected conversation prompt format ====")
        print("Expected to find substring:", repr(k))
        print("Prompt repr:", repr(prompt))
        print("Prompt raw:\n" + prompt)
        print("==== END DEBUG ====\n")
        raise ValueError("Could not find 'ASSISTANT:' in conversation prompt")
    ans = prompt[i + len(k) :].strip()
    if "</s>" in ans:
        ans = ans.split("</s>", 1)[0].strip()
    return ans


_BEGIN_STR = "<image>\n"


def _extract_body_after_image_token(q: str) -> str:
    q = str(q)
    if "<image>" in q:
        after = q.split("<image>", 1)[1]
        if "\n" in after:
            after = after.split("\n", 1)[1]
        return after.lstrip()
    return q.strip()


def _question_for_model(q: str) -> str:
    body = _extract_body_after_image_token(q)
    if body == "":
        raise ValueError("Empty question body after <image> normalization")
    return _BEGIN_STR + body


class Sa2VAEntitySegDataset(Sa2VABaseDataset):
    """
    Wrap GLaMM EntitySeg instance segmentation dataset (one instance per sample).
    Emits Sa2VA multitask samples with a STAR task only.
    """

    def __init__(
        self,
        samples_json: str,
        global_image_encoder: str = "openai/clip-vit-large-patch14-336",
        ann_train_json: str = DEFAULT_TRAIN_ANN_JSON,
        ann_val_json: str = DEFAULT_VAL_ANN_JSON,
        image_roots: Optional[list[str]] = None,
        marker_size: int = 32,
        tokenizer=None,
        prompt_template=None,
        max_length: int = 8192,
        special_tokens=None,
        arch_type: Literal["qwen"] = "qwen",
        preprocessor=None,
        extra_image_processor=None,
        repeats: float = 1.0,
        name: str = "Sa2VA_EntitySeg",
        conv_type: str = "llava_v1",
        qwen_min_pixels: int = 512 * 28 * 28,
        qwen_max_pixels: int = 2048 * 28 * 28,
    ):
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
        )
        if self.arch_type != "qwen":
            raise ValueError("Sa2VAEntitySegDataset only supports arch_type='qwen'")
        if self.preprocessor is None:
            raise ValueError("Qwen preprocessor must be provided")

        self.qwen_min_pixels = int(qwen_min_pixels)
        self.qwen_max_pixels = int(qwen_max_pixels)

        from model.llava import conversation as conversation_lib
        if conv_type not in conversation_lib.conv_templates:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_type]

        from dataset.entity_datasets.Entity_InsSeg import EntitySegDataset as _DS
        self._glamm = _DS(
            samples_json=samples_json,
            global_image_encoder=global_image_encoder,
            ann_train_json=ann_train_json,
            ann_val_json=ann_val_json,
            image_roots=image_roots,
            marker_size=marker_size,
            use_star=True,
            use_referring=False,
            use_vqa=False,
        )

    def real_len(self):
        return len(self._glamm)

    @property
    def modality_length(self):
        return [self._get_modality_length_default() for _ in range(len(self))]

    def _process_qwen_image(self, img_chw_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_chw_float = img_chw_float.clamp(0.0, 1.0)
        pil = to_pil_image(img_chw_float)
        merge_length = self.preprocessor.image_processor.merge_size ** 2
        d = self.preprocessor.image_processor(
            images=[pil],
            min_pixels=self.qwen_min_pixels,
            max_pixels=self.qwen_max_pixels,
        )
        pixel_values = torch.as_tensor(d["pixel_values"], dtype=torch.float32)
        image_grid_thw = torch.as_tensor(d["image_grid_thw"], dtype=torch.long)
        num_image_tokens = int(image_grid_thw[0].prod().item()) // int(merge_length)
        return pixel_values, image_grid_thw, num_image_tokens

    def _pack_task(
        self,
        base_idx: int,
        qwen_image: torch.Tensor,
        question: str,
        answer: str,
    ) -> _TaskPack:
        pixel_values, image_grid_thw, num_image_tokens = self._process_qwen_image(qwen_image)
        image_token_str = self._create_image_token_string(num_image_tokens)
        conv = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
        conv = self._process_conversations_for_encoding(conv, image_token_str=image_token_str, is_video=False)
        conv_prompt = conv[0]["input"] if len(conv) > 0 and "input" in conv[0] else ""
        token_dict = self.get_inputid_labels(conv)
        return _TaskPack(
            input_ids=token_dict["input_ids"],
            labels=token_dict["labels"],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            conv_prompt=conv_prompt,
            base_idx=base_idx,
        )

    def prepare_data(self, index: int) -> Optional[Dict[str, Any]]:
        ex = self._glamm[index]
        img_star = ex.get("image_star", None)
        grounding = ex.get("grounding_image", None)
        masks = ex.get("masks", None)
        if img_star is None or grounding is None or masks is None:
            raise ValueError("Expected image_star, grounding_image and masks from GLaMM EntitySeg dataset")

        if masks.ndim != 3 or masks.shape[0] != 1:
            raise ValueError(f"Expected masks of shape [1,H,W], got {tuple(masks.shape)}")

        star_q = ex.get("star", {}).get("question", None)
        star_conv = ex.get("star", {}).get("conversation", None)
        if star_q is None or star_conv is None:
            raise ValueError("Expected star question/conversation from GLaMM EntitySeg dataset")
        if not isinstance(star_conv, list) or len(star_conv) != 1:
            raise ValueError("Expected star conversation as list[str] of length 1")
        if not isinstance(star_q, str):
            raise ValueError(f"Expected star question as str, got {type(star_q)}")

        star_ans = _extract_answer_from_llava_prompt(star_conv[0])
        grounding_u8 = (grounding.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        star_q_for_model = _question_for_model(star_q)
        if not star_q_for_model.startswith("<image>"):
            raise ValueError("EntitySeg STAR question does not start with <image> after normalization")
        tp = self._pack_task(index, img_star, star_q_for_model, star_ans)
        return {
            "src": "entityseg",
            "meta": {
                "dataset": "entityseg",
                "index": int(index),
                "filepath": ex.get("filepath", None),
                "ann_id": ex.get("ann_id", None),
                "sampled_classes": ex.get("sampled_classes", None),
                "coords": ex.get("coords", None),
            },
            "images_star": img_star,
            "g_pixel_values": grounding_u8,
            "masks": masks,
            "tasks": {
                "star": {**tp.__dict__, "convs": tp.conv_prompt},
            },
        }


def _save_chw_float01(t: torch.Tensor, path: str) -> None:
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB tensor, got {tuple(t.shape)}")
    img = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img, mode="RGB").save(path)


def _save_hw_mask(m: torch.Tensor, path: str) -> None:
    if m.ndim != 2:
        raise ValueError(f"Expected HW mask, got {tuple(m.shape)}")
    img = (m > 0).to(torch.uint8).mul(255).cpu().numpy()
    Image.fromarray(img, mode="L").save(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_json", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out_dir", default="./_debug_entityseg")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer, Qwen2_5_VLProcessor
    from xtuner.utils import PROMPT_TEMPLATE

    path = "Qwen/Qwen2.5-VL-7B-Instruct"
    tok = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
        padding_side="right",
    )
    pre = dict(
        type=Qwen2_5_VLProcessor.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
    )

    ds = Sa2VAEntitySegDataset(
        samples_json=args.samples_json,
        tokenizer=tok,
        prompt_template=PROMPT_TEMPLATE.qwen_chat,
        preprocessor=pre,
        repeats=1.0,
    )

    packed = ds.prepare_data(args.idx)
    print("tasks:", list(packed["tasks"].keys()))
    print("g_pixel_values:", tuple(packed["g_pixel_values"].shape), packed["g_pixel_values"].dtype)
    print("masks:", tuple(packed["masks"].shape), packed["masks"].dtype)
    t = packed["tasks"]["star"]
    print(
        f"[star] input_ids={len(t['input_ids'])} labels={len(t['labels'])} "
        f"pixel_values={tuple(t['pixel_values'].shape)} image_grid_thw={tuple(t['image_grid_thw'].shape)}"
    )

    raw = ds._glamm[args.idx]
    if raw.get("image_star") is not None:
        _save_chw_float01(raw["image_star"], os.path.join(args.out_dir, "image_star.png"))
    if raw.get("grounding_image") is not None:
        _save_chw_float01(raw["grounding_image"], os.path.join(args.out_dir, "grounding.png"))
    if raw.get("masks") is not None:
        m = raw["masks"]
        if m.ndim == 3 and m.shape[0] > 0:
            _save_hw_mask(m[0], os.path.join(args.out_dir, "mask0.png"))

