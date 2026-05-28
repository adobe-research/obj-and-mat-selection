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

from utils.paths import material_data_root


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
        raise ValueError("Could not find 'ASSISTANT:' in conversation prompt")
    ans = prompt[i + len(k):].strip()
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


def _question_for_logging(q: str) -> str:
    return _extract_body_after_image_token(q)


class Sa2VAMaterialDataset(Sa2VABaseDataset):
    """
    Wraps GLaMM material datasets (SynmatDataset / RealmatDataset / SamaDataset)
    and emits Qwen2.5-VL-ready per-task entries while preserving GLaMM's
    sampling / crop / star / aggregate logic.
    """

    def __init__(
        self,
        source: Literal["synmat", "realmat", "sama"],
        samples_json: Optional[str] = None,
        base_data_dir: Optional[str] = None,
        merges_json: Optional[str] = None,
        description_json: Optional[str] = None,
        vqa_json: Optional[str] = None,
        image_size: tuple[int, int] = (1024, 1024),
        use_star: bool = True,
        use_referring: bool = False,
        use_vqa: bool = False,
        # Sa2VA / Qwen tokenization + processors
        tokenizer=None,
        prompt_template=None,
        max_length: int = 8192,
        special_tokens=None,
        arch_type: Literal["qwen"] = "qwen",
        preprocessor=None,
        extra_image_processor=None,
        repeats: float = 1.0,
        name: str = "Sa2VA_Material",
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
            raise ValueError("Sa2VAMaterialDataset only supports arch_type='qwen'")
        if self.preprocessor is None:
            raise ValueError("Qwen preprocessor must be provided")

        self.qwen_min_pixels = int(qwen_min_pixels)
        self.qwen_max_pixels = int(qwen_max_pixels)

        from model.llava import conversation as conversation_lib
        if conv_type not in conversation_lib.conv_templates:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_type]

        self.source = source
        base_data_dir = str(base_data_dir or material_data_root())

        ds_kwargs = dict(
            image_size=image_size,
            use_star=use_star,
            use_referring=use_referring,
            use_vqa=use_vqa,
            base_data_dir=base_data_dir,
            description_json=description_json,
            vqa_json=vqa_json,
        )
        if samples_json is not None:
            ds_kwargs["samples_json"] = samples_json

        if source == "synmat":
            from dataset.material_datasets.SynmatDataset import SynmatDataset as _DS
            if merges_json is not None:
                ds_kwargs["merges_json"] = merges_json
        elif source == "realmat":
            from dataset.material_datasets.RealmatDataset import RealmatDataset as _DS
        elif source == "sama":
            from dataset.material_datasets.SamaDataset import SamaDataset as _DS
        else:
            raise ValueError(f"Unknown source: {source!r}. Expected 'synmat', 'realmat', or 'sama'.")

        self._glamm = _DS(**ds_kwargs)

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

    def _pack_task(self, base_idx: int, qwen_image: torch.Tensor, question: str, answer: str) -> _TaskPack:
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
        img_no_star = ex.get("image_without_star", None)
        grounding = ex.get("grounding_image", None)
        masks = ex.get("masks", None)

        if grounding is None or masks is None:
            raise ValueError("Expected grounding_image and masks from GLaMM material dataset")

        grounding_u8 = (grounding.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        sampled_classes = ex.get("sampled_classes", None)
        mat_id = None
        if isinstance(sampled_classes, (list, tuple)) and len(sampled_classes) > 0:
            try:
                mat_id = int(sampled_classes[0])
            except Exception:
                pass

        out: Dict[str, Any] = {
            "src": self.source,
            "meta": {
                "dataset": "material",
                "source": self.source,
                "index": int(index),
                "filepath": ex.get("filepath", None),
                "mat_id": mat_id,
                "sampled_classes": sampled_classes,
                "coords": ex.get("coords", None),
                "aggregate": ex.get("aggregate", False),
            },
            "images_star": img_star,
            "images_without_star": img_no_star,
            "g_pixel_values": grounding_u8,
            "masks": masks,
            "tasks": {},
        }

        star_q = ex.get("star", {}).get("question", None)
        star_conv = ex.get("star", {}).get("conversation", None)
        if star_q is not None and star_conv is not None:
            if img_star is None:
                raise ValueError("STAR enabled but image_star is None")
            star_ans = _extract_answer_from_llava_prompt(star_conv[0])
            star_q_for_model = _question_for_model(star_q)
            tp = self._pack_task(index, img_star, star_q_for_model, star_ans)
            out["tasks"]["star"] = {**tp.__dict__, "convs": tp.conv_prompt}

        ref_q = ex.get("referring", {}).get("question", None)
        ref_conv = ex.get("referring", {}).get("conversation", None)
        if ref_q is not None and ref_conv is not None:
            if img_no_star is None:
                raise ValueError("Referring enabled but image_without_star is None")
            ref_ans = _extract_answer_from_llava_prompt(ref_conv[0])
            ref_q_for_model = _question_for_model(ref_q)
            tp = self._pack_task(index, img_no_star, ref_q_for_model, ref_ans)
            out["tasks"]["referring"] = {
                **tp.__dict__,
                "convs": tp.conv_prompt,
                "question": _question_for_logging(ref_q),
            }

        vqa_q = ex.get("vqa", {}).get("question", None)
        vqa_a = ex.get("vqa", {}).get("answer", None)
        if vqa_q is not None and vqa_a is not None:
            if img_star is None:
                raise ValueError("VQA enabled but image_star is None")
            vqa_q_for_model = _question_for_model(vqa_q)
            tp = self._pack_task(index, img_star, vqa_q_for_model, vqa_a)
            out["tasks"]["vqa"] = {**tp.__dict__, "convs": tp.conv_prompt}
            if isinstance(out.get("meta"), dict):
                out["meta"]["vqa_question_raw"] = vqa_q
                out["meta"]["vqa_answer_raw"] = vqa_a

        if len(out["tasks"]) == 0:
            raise ValueError("No tasks produced for this sample")

        return out


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
    # uv run python -m projects.sa2va.datasets.glamm_material --source synmat --idx 0 --out_dir ./_debug_material
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synmat", "realmat", "sama"], required=True)
    parser.add_argument("--stage", default="train")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--out_dir", default="./_debug_material")
    parser.add_argument("--no_star", action="store_true")
    parser.add_argument("--no_ref", action="store_true")
    parser.add_argument("--no_vqa", action="store_true")
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

    ds = Sa2VAMaterialDataset(
        source=args.source,
        stage=args.stage,
        use_star=not args.no_star,
        use_referring=not args.no_ref,
        use_vqa=not args.no_vqa,
        tokenizer=tok,
        prompt_template=PROMPT_TEMPLATE.qwen_chat,
        preprocessor=pre,
        repeats=1.0,
    )

    packed = ds.prepare_data(args.idx)
    print("tasks:", list(packed["tasks"].keys()))
    print("aggregate:", packed["meta"]["aggregate"])
    print("g_pixel_values:", tuple(packed["g_pixel_values"].shape), packed["g_pixel_values"].dtype)
    print("masks:", tuple(packed["masks"].shape), packed["masks"].dtype)
    for tn, t in packed["tasks"].items():
        print(
            f"[{tn}] input_ids={len(t['input_ids'])} labels={len(t['labels'])} "
            f"pixel_values={tuple(t['pixel_values'].shape)} image_grid_thw={tuple(t['image_grid_thw'].shape)}"
        )

    raw = ds._glamm[args.idx]
    if raw.get("image_star") is not None:
        _save_chw_float01(raw["image_star"], os.path.join(args.out_dir, "image_star.png"))
    if raw.get("image_without_star") is not None:
        _save_chw_float01(raw["image_without_star"], os.path.join(args.out_dir, "image_without_star.png"))
    if raw.get("grounding_image") is not None:
        _save_chw_float01(raw["grounding_image"], os.path.join(args.out_dir, "grounding.png"))
    if raw.get("masks") is not None:
        m = raw["masks"]
        if m.ndim == 3 and m.shape[0] > 0:
            _save_hw_mask(m[0], os.path.join(args.out_dir, "mask0.png"))
