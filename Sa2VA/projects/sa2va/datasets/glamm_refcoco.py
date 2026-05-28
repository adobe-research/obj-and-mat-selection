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
    """
    Normalize any prompt to "question body" (no begin_str).

    Handles both:
    - Sa2VA style: "<image>\n<question>"
    - GLaMM style: "The <image> provides ...\n<question>"
    - plain: "<question>"
    """
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


def _question_for_logging(q: str) -> str:
    return _extract_body_after_image_token(q)


class Sa2VARefCOCODataset(Sa2VABaseDataset):
    """
    Wrap GLaMM RefCOCO-family referring segmentation dataset and emit Sa2VA multitask samples.
    Only supports num_classes_per_sample=1.
    """

    def __init__(
        self,
        dataset_dir: str,
        refer_segm_data: str = "refcoco||refcoco+||refcocog",
        split: str = "val",
        image_size: int = 1024,
        num_classes_per_sample: int = 1,
        global_image_encoder: str = "openai/clip-vit-large-patch14-336",
        src_name: Optional[str] = None,
        tokenizer=None,
        prompt_template=None,
        max_length: int = 8192,
        special_tokens=None,
        arch_type: Literal["qwen"] = "qwen",
        preprocessor=None,
        extra_image_processor=None,
        repeats: float = 1.0,
        name: str = "Sa2VA_RefCOCO",
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
            raise ValueError("Sa2VARefCOCODataset only supports arch_type='qwen'")
        if self.preprocessor is None:
            raise ValueError("Qwen preprocessor must be provided")
        if int(num_classes_per_sample) != 1:
            raise ValueError("Sa2VARefCOCODataset requires num_classes_per_sample=1")

        self.src_name = str(src_name) if src_name is not None else "refcoco"
        self.split = str(split)
        self.refer_segm_data = str(refer_segm_data)

        self.qwen_min_pixels = int(qwen_min_pixels)
        self.qwen_max_pixels = int(qwen_max_pixels)

        from model.llava import conversation as conversation_lib
        if conv_type not in conversation_lib.conv_templates:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_type]

        from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset as _DS
        self._glamm = _DS(
            dataset_dir=dataset_dir,
            global_image_encoder=global_image_encoder,
            image_size=image_size,
            num_classes_per_sample=1,
            refer_segm_data=refer_segm_data,
            split=split,
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
        grounding = ex.get("grounding_image", None)
        masks = ex.get("masks", None)
        if grounding is None or masks is None:
            raise ValueError("Expected grounding_image and masks from GLaMM RefCOCO dataset")

        if masks.ndim != 3 or masks.shape[0] != 1:
            raise ValueError(f"Expected masks of shape [1,H,W], got {tuple(masks.shape)}")

        ref_q = ex.get("referring", {}).get("question", None)
        ref_conv = ex.get("referring", {}).get("conversation", None)
        if ref_q is None or ref_conv is None:
            raise ValueError("Expected referring question/conversation from GLaMM RefCOCO dataset")
        if not isinstance(ref_conv, list) or len(ref_conv) != 1:
            raise ValueError("Expected referring conversation as list[str] of length 1")
        if isinstance(ref_q, list):
            if len(ref_q) != 1:
                raise ValueError("Expected one question when num_classes_per_sample=1")
            ref_q = ref_q[0]
        if not isinstance(ref_q, str):
            raise ValueError(f"Expected referring question as str, got {type(ref_q)}")

        ref_q_raw = ref_q
        ref_q_for_model = _question_for_model(ref_q_raw)
        if not ref_q_for_model.startswith("<image>"):
            raise ValueError("RefCOCO question does not start with <image> after normalization")

        ref_ans = _extract_answer_from_llava_prompt(ref_conv[0])
        grounding_u8 = (grounding.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        img_no_star = ex.get("image_without_star", None)
        if img_no_star is None:
            raise ValueError("Expected image_without_star from GLaMM RefCOCO dataset")

        tp = self._pack_task(index, img_no_star, ref_q_for_model, ref_ans)
        return {
            "src": self.src_name,
            "meta": {
                "dataset": self.src_name,
                "index": int(index),
                "filepath": ex.get("filepath", None),
                "ann_id": ex.get("ann_id", None),
                "sampled_classes": ex.get("sampled_classes", None),
                "split": self.split,
                "refer_segm_data": self.refer_segm_data,
            },
            "images_without_star": img_no_star,
            "g_pixel_values": grounding_u8,
            "masks": masks,
            "tasks": {
                "referring": {
                    **tp.__dict__,
                    "convs": tp.conv_prompt,
                    "question": _question_for_logging(ref_q_raw),
                },
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
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out_dir", default="./_debug_refcoco")
    parser.add_argument("--refer_segm_data", default="refcoco")
    parser.add_argument("--split", default="train")
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

    ds = Sa2VARefCOCODataset(
        dataset_dir=args.dataset_dir,
        refer_segm_data=args.refer_segm_data,
        split=args.split,
        tokenizer=tok,
        prompt_template=PROMPT_TEMPLATE.qwen_chat,
        preprocessor=pre,
        repeats=1.0,
    )

    packed = ds.prepare_data(args.idx)
    print("tasks:", list(packed["tasks"].keys()))
    print("g_pixel_values:", tuple(packed["g_pixel_values"].shape), packed["g_pixel_values"].dtype)
    print("masks:", tuple(packed["masks"].shape), packed["masks"].dtype)
    t = packed["tasks"]["referring"]
    print(
        f"[referring] input_ids={len(t['input_ids'])} labels={len(t['labels'])} "
        f"pixel_values={tuple(t['pixel_values'].shape)} image_grid_thw={tuple(t['image_grid_thw'].shape)}"
    )

    raw = ds._glamm[args.idx]
    if raw.get("image_without_star") is not None:
        _save_chw_float01(raw["image_without_star"], os.path.join(args.out_dir, "image_without_star.png"))
    if raw.get("grounding_image") is not None:
        _save_chw_float01(raw["grounding_image"], os.path.join(args.out_dir, "grounding.png"))
    if raw.get("masks") is not None:
        m = raw["masks"]
        if m.ndim == 3 and m.shape[0] > 0:
            _save_hw_mask(m[0], os.path.join(args.out_dir, "mask0.png"))

