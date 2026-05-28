import os
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image as PILImage
from typing import Optional

from transformers import CLIPImageProcessor

_GLAMM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_GLAMM_ROOT, ".."))
for _p in (_GLAMM_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.glamm_eval_utils import DEFAULT_IMAGE_TOKEN
from model.llava import conversation as conversation_lib
from dataset.utils.utils import ANSWER_LIST, VQA_QUESTIONS, STAR_QUESTIONS, REFERRING_QUESTIONS
from utils.hm_utils import augment_data, add_star_marker


class BaseMaterialDataset(torch.utils.data.Dataset):
    """
    Eval-only base class for material segmentation datasets.
    Release JSON: flat list [{source, filepath, mat_id, ...}]
    Images/masks at {base_data_dir}/{source}/{images,masks}/{stem}.png
    """

    def __init__(
        self,
        samples_json: str,
        base_data_dir: str,
        description_json: Optional[str] = None,
        vqa_json: Optional[str] = None,
        image_size=(1024, 1024),
        use_star: bool = True,
        use_vqa: bool = False,
        use_referring: bool = False,
        marker_size: int = 32,
    ):
        super().__init__()
        self.image_size = image_size
        self.base_data_dir = base_data_dir
        self.marker_size = marker_size
        self.use_star = use_star
        self.use_referring = use_referring
        self.use_vqa = use_vqa

        with open(samples_json) as f:
            data = json.load(f)
        self.samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
        print(f"Loaded {len(self.samples)} samples from {samples_json}")

        if self.use_referring:
            if description_json is None:
                raise ValueError("description_json required when use_referring=True")
            with open(description_json) as f:
                self.description = json.load(f)

        if self.use_vqa:
            if vqa_json is None:
                raise ValueError("vqa_json required when use_vqa=True")
            with open(vqa_json) as f:
                raw = json.load(f)
            self.vqa = raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw

        self.star_questions = STAR_QUESTIONS
        self.referring_questions = REFERRING_QUESTIONS
        self.vqa_questions = VQA_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.global_image_encoder = "openai/clip-vit-large-patch14-336"
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(self.global_image_encoder)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filepath_to_stem(filepath: str) -> str:
        """'/synmat/AI09_002_frame0780_sel.exr' -> 'AI09_002_frame0780_sel'"""
        rel = filepath.lstrip("/").split("/", 1)[1]
        parts = rel.replace("\\", "/").split("/")
        parts[-1] = os.path.splitext(parts[-1])[0]
        return "__".join(parts)

    def _lookup_key(self, filepath: str) -> str:
        """Key used to look up this filepath in description/VQA JSONs.
        Default: basename. Override in subclasses where keys differ (e.g. realmat)."""
        return os.path.basename(filepath)

    def _load_image_and_label(self, sample, filepath):
        source = sample["source"]
        mat_id = int(sample["mat_id"])
        stem = self._filepath_to_stem(sample["filepath"])

        img_path = os.path.join(self.base_data_dir, source, "images", f"{stem}.png")
        msk_path = os.path.join(self.base_data_dir, source, "masks", f"{stem}_mat{mat_id}.png")

        pil = PILImage.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(pil)).float().div(255.0).permute(2, 0, 1)
        h, w = image.shape[1], image.shape[2]

        mat_label = torch.zeros(1, h, w, dtype=torch.long)
        if os.path.exists(msk_path):
            mask_arr = np.array(PILImage.open(msk_path).convert("L")) > 127
            mat_label[0] = torch.from_numpy(mask_arr).long() * mat_id

        # Fixed resize to image_size shortest side (no random scale for eval)
        image = transforms.Resize(self.image_size[0])(image)
        mat_label = transforms.Resize(self.image_size[0], interpolation=InterpolationMode.NEAREST_EXACT)(mat_label)

        return image, mat_label

    def _create_conversation(self, question, answer):
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        return [conv.get_prompt()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        filepath = self.base_data_dir + sample["filepath"]

        image, mat_label = self._load_image_and_label(sample, filepath)

        target_mat_id = int(sample["mat_id"])
        aggregate = bool(sample.get("aggregate", False))

        mask = mat_label[0] == target_mat_id
        if mask.sum() == 0:
            raise ValueError(f"Empty mask at index {index}, mat_id={target_mat_id}, file={sample['filepath']}")

        image, mat_label = augment_data(
            image, mat_label,
            size=self.image_size,
            flip=False,
            test=True,
            crop=False,
            resize=False,
            point_hw=None,
        )

        mask = mat_label[0] == target_mat_id
        if mask.sum() == 0:
            raise ValueError(f"Empty mask after augment at index {index}")

        masks = mask.float().unsqueeze(0)
        original_h, original_w = image.shape[-2:]

        # Deterministic star placement: candidate closest to centroid of valid region.
        # Falls back through progressively smaller kernels down to k=1 (any mask pixel).
        invalid = (~mask).to(torch.float32).unsqueeze(0).unsqueeze(0)
        ys_8, xs_8 = torch.tensor([]), torch.tensor([])
        r_8 = 0
        for k in (self.marker_size // 4, self.marker_size // 8, 3, 1):
            if k < 1:
                continue
            bad = F.max_pool2d(invalid, kernel_size=k, stride=1, padding=0)[0, 0]
            ys_8, xs_8 = torch.where(bad == 0)
            if len(ys_8) > 0:
                r_8 = k // 2
                break

        if len(ys_8) == 0:
            raise ValueError(f"No valid star position at index {index}")

        cent_h = ys_8.float().mean().round().long()
        cent_w = xs_8.float().mean().round().long()
        dists = (ys_8 - cent_h).abs() + (xs_8 - cent_w).abs()
        best = dists.argmin()
        rand_h = int(ys_8[best].item()) + r_8
        rand_w = int(xs_8[best].item()) + r_8

        rand_h_list = torch.tensor([rand_h])
        rand_w_list = torch.tensor([rand_w])

        image_for_sam = image.clone() if (self.use_star or self.use_referring) else None
        image_without_star = image.clone() if self.use_referring else None

        begin_str = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"
        first_color = None

        if self.use_star or self.use_vqa:
            if not mask[rand_h, rand_w]:
                raise ValueError(f"Star pixel not on mask at index {index}")
            image, first_color = add_star_marker(image, rand_h, rand_w, size=self.marker_size, color=None)

        # STAR
        star_question = star_answer = star_conv_prompt = None
        if self.use_star and first_color is not None:
            star_question = begin_str + self.star_questions[0].replace("<COLOR>", first_color)
            star_answer = self.answer_list[0]
            star_conv_prompt = self._create_conversation(star_question, star_answer)

        # REFERRING
        referring_question = referring_answer = referring_conv_prompt = desc = None
        if self.use_referring:
            key = str(target_mat_id)
            desc_file_key = self._lookup_key(sample["filepath"])
            if desc_file_key not in self.description or key not in self.description[desc_file_key]:
                raise KeyError(f"Description missing for {desc_file_key} mat_id={key}")
            descriptions = self.description[desc_file_key][key].get("descriptions")
            if not descriptions:
                raise ValueError(f"Empty descriptions for {desc_file_key} mat_id={key}")
            desc = descriptions[0].lower()
            referring_question = begin_str + self.referring_questions[0].replace("<DESCRIPTION>", desc)
            referring_answer = self.answer_list[0]
            referring_conv_prompt = self._create_conversation(referring_question, referring_answer)

        # VQA
        vqa_question = vqa_answer = vqa_conv_prompt = None
        if self.use_vqa:
            vqa_key = self._lookup_key(sample["filepath"])
            if vqa_key not in self.vqa:
                raise KeyError(f"{vqa_key} missing in VQA samples")
            mat_key = str(target_mat_id)
            if mat_key not in self.vqa[vqa_key]:
                raise KeyError(f"mat_id {mat_key} missing for {vqa_key}")
            vqa_list = self.vqa[vqa_key][mat_key]
            if not vqa_list:
                raise ValueError(f"Empty VQA list for {vqa_key} mat_id={mat_key}")
            opt = vqa_list[0]
            q = begin_str + self.vqa_questions[0].replace("<COLOR>", first_color or "blue") + "\n\n"
            for choice in ["A", "B", "C", "D"]:
                q += f"{choice}. {opt[choice]}\n"
            q += "Answer directly with the option letter from the given choices."
            vqa_question = q
            vqa_answer = opt["answer"] + "."
            vqa_conv_prompt = self._create_conversation(q, vqa_answer)

        coords = (
            torch.stack([rand_h_list, rand_w_list], dim=1).tolist()
            if (self.use_star or self.use_vqa or self.use_referring) else []
        )

        return {
            "filepath": filepath,
            "image_star": image if (self.use_star or self.use_vqa) else None,
            "image_without_star": image_without_star,
            "grounding_image": image_for_sam,
            "global_enc_processor": self.global_enc_processor,
            "masks": masks,
            "orig_size": (original_h, original_w),
            "sampled_classes": [target_mat_id],
            "coords": coords,
            "aggregate": aggregate,
            "star": {"question": star_question, "conversation": star_conv_prompt},
            "referring": {"question": referring_question, "conversation": referring_conv_prompt, "desc": desc},
            "vqa": {"question": vqa_question, "answer": vqa_answer, "conversation": vqa_conv_prompt},
        }
