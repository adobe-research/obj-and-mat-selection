import os
import sys
import json
import torch
import argparse
import numpy as np
import gradio as gr
import transformers
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import io
import zipfile
import tempfile
from datetime import datetime


from PIL import Image
from matplotlib.colors import Normalize
from transformers import CLIPImageProcessor

MMSEG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GLAMM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MMSEG_ROOT)
sys.path.insert(0, GLAMM_ROOT)

from functools import partial
from model.GLaMM import GLaMMForCausalLM
from utils.hm_utils import add_star_marker, create_mask_overlay
from dataset.datasets import custom_collate_fn_multi
from model.llava import conversation as conversation_lib
from dataset.utils.utils import STAR_QUESTIONS, REFERRING_QUESTIONS, SEG_QUESTIONS, TASK_PROMPT
from tools.glamm_eval_utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    dict_to_cuda,
    intersectionAndUnionGPU,
)
from check_load import load_mp_rank_checkpoint


class GLaMMDemo:
    def __init__(self, model_path, args_dict):
        print(f"[DEBUG] __init__: Model path: {model_path}")
        self.model_path = model_path
        self.args_dict = args_dict
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEBUG] __init__: Using device: {self.device}")

        self._setup_tokenizer()
        self._setup_model()

    @staticmethod
    def _ensure_unit_range(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor values fall in [0, 1], matching PNG-like range."""
        if tensor.numel() == 0:
            return tensor

        tensor = tensor.to(dtype=torch.float32)
        t_min = float(tensor.min().item())
        t_max = float(tensor.max().item())

        if 0.0 <= t_min and t_max <= 1.0:
            return tensor

        if 0.0 <= t_min and t_max <= 255.0:
            tensor = tensor / 255.0
        else:
            denom = max(t_max - t_min, 1e-6)
            tensor = (tensor - t_min) / denom

        return tensor.clamp_(0.0, 1.0)

    def _image_to_tensor(
        self, image: torch.Tensor | np.ndarray | Image.Image
    ) -> torch.Tensor:
        """Convert various image inputs to CHW float tensor in [0, 1]."""
        if isinstance(image, torch.Tensor):
            image_tensor = image.detach().clone()
            if image_tensor.ndim == 3 and image_tensor.shape[0] in (1, 3):
                pass
            elif image_tensor.ndim == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            elif image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=torch.float32)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
                image_tensor = transforms.ToTensor()(pil_image)
            else:
                image_tensor = torch.from_numpy(image).float()
                if image_tensor.ndim == 3 and image_tensor.shape[2] in (1, 3):
                    image_tensor = image_tensor.permute(2, 0, 1)
                elif image_tensor.ndim == 2:
                    image_tensor = image_tensor.unsqueeze(0)
                image_tensor = (
                    image_tensor / 255.0 if image_tensor.max() > 1.0 else image_tensor
                )
        else:
            image_tensor = transforms.ToTensor()(image)

        return self._ensure_unit_range(image_tensor)

    def _setup_tokenizer(self):
        """Setup tokenizer following eval_vis.py logic exactly."""
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.args_dict.get("model_max_length", 1536),
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        if not self.args_dict.get("pretrained", False):
            if self.args_dict.get("use_mm_start_end", True):
                self.tokenizer.add_tokens(
                    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
                )
            reg_tokens = ["<bbox>", "<point>"]
            segmentation_tokens = ["[SEG]"]
            phrase_tokens = ["<p>", "</p>"]
            special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        print(f"Token 29871 decodes to: '{self.tokenizer.decode([29871])}'")
        print(f"Token 32004 decodes to: '{self.tokenizer.decode([32004])}'")

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bbox>", "[SEG]", "<p>", "</p>"]}
        )
        print("Special tokens", self.tokenizer.additional_special_tokens)

        self.bbox_token_idx = self.tokenizer(
            "<bbox>", add_special_tokens=False
        ).input_ids[0]
        self.seg_token_idx = self.tokenizer(
            "[SEG]", add_special_tokens=False
        ).input_ids[0]
        self.bop_token_idx = self.tokenizer("<p>", add_special_tokens=False).input_ids[
            0
        ]
        self.eop_token_idx = self.tokenizer("</p>", add_special_tokens=False).input_ids[
            0
        ]

        print(f"[DEBUG] _setup_tokenizer: Special token indices:")
        print(f"[DEBUG] _setup_tokenizer: bbox_token_idx: {self.bbox_token_idx}")
        print(f"[DEBUG] _setup_tokenizer: seg_token_idx: {self.seg_token_idx}")
        print(f"[DEBUG] _setup_tokenizer: bop_token_idx: {self.bop_token_idx}")
        print(f"[DEBUG] _setup_tokenizer: eop_token_idx: {self.eop_token_idx}")
        print(
            f"[DEBUG] _setup_tokenizer: Tokenizer pad token: {self.tokenizer.pad_token}"
        )
        print(f"[DEBUG] _setup_tokenizer: Tokenizer vocab size: {len(self.tokenizer)}")

    def _setup_model(self):
        """Setup model following eval_vis.py logic exactly."""
        model_args = {
            "train_mask_decoder": self.args_dict.get("train_mask_decoder", True),
            "out_dim": self.args_dict.get("out_dim", 256),
            "ce_loss_weight": self.args_dict.get("ce_loss_weight", 1.0),
            "dice_loss_weight": self.args_dict.get("dice_loss_weight", 0.5),
            "bce_loss_weight": self.args_dict.get("bce_loss_weight", 2.0),
            "seg_token_idx": self.seg_token_idx,
            "vision_pretrained": self.args_dict.get(
                "vision_pretrained",
                os.environ.get("SAM_CKPT", ""),
            ),
            "vision_tower": self.args_dict.get(
                "vision_tower", "openai/clip-vit-large-patch14-336"
            ),
            "use_mm_start_end": self.args_dict.get("use_mm_start_end", True),
            "mm_vision_select_layer": self.args_dict.get("mm_vision_select_layer", -2),
            "pretrain_mm_mlp_adapter": self.args_dict.get(
                "pretrain_mm_mlp_adapter", ""
            ),
            "tune_mm_mlp_adapter": self.args_dict.get("tune_mm_mlp_adapter", False),
            "freeze_mm_mlp_adapter": self.args_dict.get("freeze_mm_mlp_adapter", False),
            "mm_use_im_start_end": self.args_dict.get("mm_use_im_start_end", True),
            "with_region": self.args_dict.get("with_region", True),
            "bbox_token_idx": self.bbox_token_idx,
            "eop_token_idx": self.eop_token_idx,
            "bop_token_idx": self.bop_token_idx,
        }
        model_args["num_level_reg_features"] = 4

        self.model = GLaMMForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **model_args,
        )

        resume_ckpt = self.args_dict.get("resume") or ""
        if resume_ckpt:
            if not os.path.exists(resume_ckpt):
                raise FileNotFoundError(
                    f"Checkpoint not found at '{resume_ckpt}'. Please provide a valid .pt file."
                )
            print(f"[DEBUG] _setup_model: Loading checkpoint from {resume_ckpt}")
            _, state_dict = load_mp_rank_checkpoint(resume_ckpt)
            cleaned_state = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            missing, unexpected = self.model.load_state_dict(
                cleaned_state, strict=False
            )
            print(
                f"[DEBUG] _setup_model: Checkpoint load complete: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16, device=self.device)

        if not self.args_dict.get("pretrained", False):
            self.model.get_model().initialize_glamm_model(self.model.get_model().config)
        else:
            for param in self.model.get_model().grounding_encoder.parameters():
                param.requires_grad = False
            if self.model.get_model().config.train_mask_decoder:
                self.model.get_model().grounding_encoder.mask_decoder.train()
                for (
                    p
                ) in self.model.get_model().grounding_encoder.mask_decoder.parameters():
                    p.requires_grad = True

            self.model.get_model().text_hidden_fcs.train()
            for p in self.model.get_model().text_hidden_fcs.parameters():
                param_requires_grad = True
                p.requires_grad = param_requires_grad

        for p in vision_tower.parameters():
            p.requires_grad = False
        for p in self.model.get_model().mm_projector.parameters():
            p.requires_grad = False

        lora_r = self.args_dict.get("lora_r", 0)
        if lora_r == 0:
            for p in self.model.get_model().layers.parameters():
                p.requires_grad = True
            for p in self.model.get_model().mm_projector.parameters():
                p.requires_grad = True

        conversation_lib.default_conversation = conversation_lib.conv_templates[
            self.args_dict.get("conv_type", "llava_v1")
        ]

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

        self.global_enc_processor = CLIPImageProcessor.from_pretrained(
            self.args_dict.get("vision_tower", "openai/clip-vit-large-patch14-336")
        )

        self.img_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.img_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        print(f"[DEBUG] _setup_model: Model loaded successfully")
        print(
            f"[DEBUG] _setup_model: Model device: {next(self.model.parameters()).device}"
        )
        print(
            f"[DEBUG] _setup_model: Model dtype: {next(self.model.parameters()).dtype}"
        )
        print(f"[DEBUG] _setup_model: Tokenizer vocab size: {len(self.tokenizer)}")
        print(
            f"[DEBUG] _setup_model: Global encoder processor: {type(self.global_enc_processor)}"
        )

    def grounding_enc_processor(
        self, x: torch.Tensor, image_size: tuple
    ) -> torch.Tensor:
        """Process image for grounding encoder following HierarchicalSyntheticDataset."""
        img_mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(
            1, -1, 1, 1
        )
        img_std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(
            1, -1, 1, 1
        )
        x = (x - img_mean) / img_std
        h, w = x.shape[-2:]
        target_h, target_w = image_size
        x = F.pad(x, (0, target_w - w, 0, target_h - h))
        return x

    def resize_image_to_square(self, image, target_size=1024):
        """Resize image to square target size using resize + center crop."""
        print(f"[DEBUG] resize_image_to_square: Input image type: {type(image)}")
        image_tensor = self._image_to_tensor(image)

        H, W = image_tensor.shape[-2:]
        if H < W:
            new_h = target_size
            new_w = int(target_size * W / H)
        else:
            new_w = target_size
            new_h = int(target_size * H / W)

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        image_tensor = transforms.CenterCrop((target_size, target_size))(image_tensor)
        return image_tensor

    def map_coordinates_to_resized(
        self, original_coords, original_size, target_size=1024
    ):
        """Map coordinates from original image to resized square image."""
        orig_h, orig_w = original_size
        click_h, click_w = original_coords
        if orig_h < orig_w:
            new_h = target_size
            new_w = int(target_size * orig_w / orig_h)
        else:
            new_w = target_size
            new_h = int(target_size * orig_h / orig_w)

        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        resized_h = int(click_h * scale_h)
        resized_w = int(click_w * scale_w)
        crop_top = (new_h - target_size) // 2
        crop_left = (new_w - target_size) // 2
        final_h = max(0, min(target_size - 1, resized_h - crop_top))
        final_w = max(0, min(target_size - 1, resized_w - crop_left))
        print(
            "[DEBUG] map_coordinates_to_resized: "
            f"orig_size={original_size}, orig_coords={original_coords}, "
            f"scaled=(h={resized_h}, w={resized_w}), crop=(top={crop_top}, left={crop_left}), "
            f"final=(h={final_h}, w={final_w})"
        )
        return (final_h, final_w)

    def _overlay_all_stars_1024(
        self, base_tensor_1024_chw: torch.Tensor, coords_1024, fixed_color: str | None
    ):
        """
        Overlays all stars on a CHW [0,1] 1024 tensor using add_star_marker ONLY.
        Returns (tensor_with_stars, latched_color).
        """
        img = base_tensor_1024_chw.clone()
        latched = fixed_color
        marker_size = max(8, int(1024 // 32))
        for i, (hh, ww) in enumerate(coords_1024):
            h = int(max(0, min(1023, hh)))
            w = int(max(0, min(1023, ww)))
            print(
                f"[DEBUG] _overlay_all_stars_1024: star #{i+1} clamped coords -> (h={h}, w={w})"
            )
            try:
                if i == 0 and latched is None:
                    img, c = add_star_marker(img, h, w, size=marker_size)
                    latched = c or "blue"
                else:
                    img, _ = add_star_marker(img, h, w, size=marker_size, color=latched)
            except TypeError:
                img, c = add_star_marker(img, h, w, size=marker_size)
                if i == 0 and latched is None:
                    latched = c or "blue"
        return img, latched

    def create_data_batch(
        self,
        image,  # PIL or np
        coords_list_1024,  # list[(h,w)] in 1024 space; [] => referring
        text_prompt="Please segment all pixels with the same material as where the <COLOR> star is.",
        fixed_star_color=None,  # first color (latched)
    ):
        """Create data batch for star/referring using ONLY add_star_marker for overlays."""
        if isinstance(image, np.ndarray):
            original_image_pil = Image.fromarray(image)
        else:
            original_image_pil = image
        orig_w, orig_h = original_image_pil.size
        print(
            f"[DEBUG] create_data_batch: original image size (W,H)=({orig_w},{orig_h})"
        )

        base_1024_chw = self.resize_image_to_square(original_image_pil, 1024)
        grounding_enc_image = base_1024_chw.clone()  # clean copy for SAM
        global_enc_tensor = base_1024_chw.clone()  # working copy for CLIP / viz

        if len(coords_list_1024) > 0:
            global_enc_tensor, latched_color = self._overlay_all_stars_1024(
                global_enc_tensor, coords_list_1024, fixed_star_color
            )
            star_color = latched_color or fixed_star_color or "blue"
        else:
            star_color = fixed_star_color

        global_enc_image = transforms.ToPILImage()(global_enc_tensor)
        print(
            "[DEBUG] create_data_batch: coords (1024 space)="
            f"{coords_list_1024}, star_color={star_color}"
        )

        if star_color is not None:
            processed_prompt = text_prompt.replace("<COLOR>", star_color)
        else:
            processed_prompt = (
                text_prompt.replace("<COLOR>", "").replace("  ", " ").strip()
            )

        print(
            f"[DEBUG] create_data_batch: Prompt after <COLOR> substitution: {processed_prompt}"
        )

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        begin_str = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"
        question = begin_str + processed_prompt
        answer = "[SEG]"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversation_str = conv.get_prompt()

        fake_batch_item = {
            "filepath": "demo_image_path",
            "image_star": global_enc_tensor,  # CHW tensor already with/without stars
            "image_without_star": grounding_enc_image.clone(),  # clean tensor
            "grounding_image": grounding_enc_image.clone(),  # SAM/grounding input
            "masks": torch.zeros(1, 1024, 1024),  # dummy for inference
            "orig_size": (1024, 1024),
            "sampled_classes": [0],
            "coords": [list(map(int, xy)) for xy in coords_list_1024],
            "global_enc_processor": self.global_enc_processor,
            "star": {
                "conversation": (
                    [conversation_str] if len(coords_list_1024) >= 1 else None
                ),
                "question": question if len(coords_list_1024) >= 1 else None,
            },
            "referring": {
                "conversation": (
                    [conversation_str] if len(coords_list_1024) == 0 else None
                ),
                "question": question if len(coords_list_1024) == 0 else None,
                "desc": processed_prompt if len(coords_list_1024) == 0 else None,
            },
            "vqa": {"conversation": None, "question": None, "answer": None},
        }

        batch_data = custom_collate_fn_multi(
            [fake_batch_item],
            tokenizer=self.tokenizer,
            use_mm_start_end=self.args_dict.get("use_mm_start_end", True),
            inference=True,
        )

        def move_to_device_recursive(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: move_to_device_recursive(v, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device_recursive(item, device) for item in obj]
            else:
                return obj

        data_batch = move_to_device_recursive(batch_data, self.device)

        if "grounding_enc_images" in data_batch:
            grounding_tensors = (
                (data_batch["grounding_enc_images"] * 255).to(torch.uint8).contiguous()
            )
            data_batch["grounding_enc_images"] = self.grounding_enc_processor(
                grounding_tensors.float(), data_batch.get("orig_size", (1024, 1024))
            )
            print(
                f"[DEBUG] create_data_batch: Processed grounding_enc_images (x255, normalize), shape: {data_batch['grounding_enc_images'].shape}, dtype: {data_batch['grounding_enc_images'].dtype}"
            )

        for key in ["global_enc_images", "images_star", "images_without_star"]:
            if (
                key in data_batch
                and isinstance(data_batch[key], torch.Tensor)
                and data_batch[key].dtype != torch.bfloat16
            ):
                data_batch[key] = data_batch[key].to(dtype=torch.bfloat16)
                print(f"[DEBUG] create_data_batch: Converted {key} to bfloat16")

        if "masks_list" in data_batch and isinstance(data_batch["masks_list"], list):
            data_batch["masks_list"] = [
                mask.to(dtype=torch.bfloat16) if mask.dtype != torch.bfloat16 else mask
                for mask in data_batch["masks_list"]
            ]
            print(f"[DEBUG] create_data_batch: Converted masks_list to bfloat16")

        viz_tensor = global_enc_tensor.clone()
        return data_batch, viz_tensor, star_color, global_enc_image

    def inference(self, image, coords_list_1024, text_prompt, fixed_star_color=None):
        """Run inference; routes to star vs referring by number of coords."""
        try:
            data_batch, image_tensor, star_color, global_enc_image = (
                self.create_data_batch(
                    image,
                    coords_list_1024,
                    text_prompt,
                    fixed_star_color=fixed_star_color,
                )
            )

            with torch.no_grad():
                results = self.model(**data_batch)

            task_key = "referring" if len(coords_list_1024) == 0 else "star"
            task_out = results.get(task_key)
            if task_out is None:
                return None, None, None, None, f"No {task_key} task results."

            predictions = task_out.get("pred_masks")
            if predictions is None:
                return (
                    None,
                    None,
                    None,
                    None,
                    f"No predictions for {task_key}. Available keys: {list(task_out.keys())}",
                )

            pred_mask_raw = predictions[0].detach().cpu().numpy()
            if pred_mask_raw.ndim == 3 and pred_mask_raw.shape[0] == 1:
                pred_mask_raw = pred_mask_raw[0]

            pred_mask_binary = (pred_mask_raw > 0).astype(np.uint8) * 255
            return (
                pred_mask_binary,
                pred_mask_raw,
                image_tensor,
                star_color,
                global_enc_image,
            )

        except Exception as e:
            import traceback

            error_msg = f"Error during inference: {str(e)}\n{traceback.format_exc()}"
            print(f"[DEBUG] inference: {error_msg}")
            return None, None, None, None, error_msg

    def create_visualization(
        self, image_1024_with_star, pred_mask, global_enc_image_pil=None
    ):
        """Create visualization showing 336->1024 global encoder image and segmentation mask."""
        if pred_mask is None:
            return None

        if pred_mask.ndim == 3:
            if pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]
            elif pred_mask.shape[-1] == 1:
                pred_mask = pred_mask[:, :, 0]

        global_enc_336_1024 = None
        clip_input_pil = None
        if image_1024_with_star is not None:
            if isinstance(image_1024_with_star, torch.Tensor):
                clip_input_pil = transforms.ToPILImage()(
                    image_1024_with_star.detach().cpu()
                )
            else:
                clip_input_pil = image_1024_with_star
        elif global_enc_image_pil is not None:
            clip_input_pil = global_enc_image_pil

        if clip_input_pil is not None:
            processed_336 = self.global_enc_processor(
                clip_input_pil, return_tensors="pt"
            )
            if "pixel_values" in processed_336:
                img_336 = processed_336["pixel_values"][0]  # [C, H, W]
                if img_336.min() < 0:
                    img_336 = (img_336 + 1.0) / 2.0
                elif img_336.max() > 1.0:
                    img_336 = img_336 / 255.0
                img_336_1024 = F.interpolate(
                    img_336.unsqueeze(0),
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                global_enc_336_1024 = img_336_1024.permute(1, 2, 0).cpu().numpy()
                global_enc_336_1024 = np.clip(global_enc_336_1024, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        if global_enc_336_1024 is not None:
            if global_enc_336_1024.ndim == 2:
                axes[0].imshow(global_enc_336_1024, cmap="gray")
            else:
                axes[0].imshow(global_enc_336_1024)
            axes[0].set_title("Global Encoder Input (336→1024)", pad=20)
        else:
            axes[0].text(0.5, 0.5, "No global encoder image", ha="center", va="center")
            axes[0].set_title("Global Encoder Input (336→1024)", pad=20)
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Segmentation (1024)", pad=20)
        axes[1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return Image.fromarray(img_array)


def create_gradio_interface(demo_model, model_title="GLaMM"):
    """Create Gradio interface with multi-click + submit (add_star_marker only)."""

    MAX_STARS = 5

    _SELECTION_MODES = {
        "Material: click": (
            STAR_QUESTIONS[0],
            "Place one or more stars on the image, then click **Submit**. `<COLOR>` is filled automatically.",
        ),
        "Material: text": (
            REFERRING_QUESTIONS[0],
            "Replace **`<DESCRIPTION>`** with your material description (e.g. *shiny chrome metal*). No stars needed.",
        ),
        "Material: click + text": (
            f"Please segment all pixels made of the material described below, where the <COLOR> star is.\nDescription: <DESCRIPTION>\n\n{TASK_PROMPT}",
            "Place a star, then replace **`<DESCRIPTION>`** with your material description.",
        ),
        "Object: text": (
            SEG_QUESTIONS[0].replace("{class_name}", "<OBJECT>"),
            "Replace **`<OBJECT>`** with your object expression (e.g. *the man in a red shirt*). No stars needed.",
        ),
    }
    _DEFAULT_MODE = "Material: click"

    def on_selection_change(mode):
        prompt, hint = _SELECTION_MODES.get(mode, _SELECTION_MODES[_DEFAULT_MODE])
        return prompt, hint

    REFERRING_DESC_EXAMPLES = [
        "a shiny chrome metal",
        "a rough concrete wall",
        "a soft beige fabric",
    ]

    def _prompt_examples_markdown() -> str:
        star_examples = []
        for i in range(min(2, len(STAR_QUESTIONS))):
            star_examples.append(f"- STAR example {i+1}:\n```\n{STAR_QUESTIONS[i]}\n```")

        referring_examples = []
        for i in range(min(2, len(REFERRING_QUESTIONS))):
            desc = REFERRING_DESC_EXAMPLES[i % len(REFERRING_DESC_EXAMPLES)]
            referring_examples.append(
                "- REFERRING example {idx}:\n```\n{txt}\n```".format(
                    idx=i + 1, txt=REFERRING_QUESTIONS[i].replace("<DESCRIPTION>", desc)
                )
            )

        refcoco_phrases = [
            "the man in a red shirt",
            "the dog on the left",
        ]
        object_examples = []
        for i in range(min(2, len(SEG_QUESTIONS))):
            phrase = refcoco_phrases[i % len(refcoco_phrases)]
            try:
                txt = SEG_QUESTIONS[i].format(class_name=phrase)
            except Exception:
                txt = SEG_QUESTIONS[i]
            object_examples.append(f"- OBJECT (RefCOCO) example {i+1}:\n```\n{txt}\n```")

        return (
            "**Prompt examples (copy/paste)**\n\n"
            "**STAR (use 1–5 stars)**: keep `<COLOR>`; it will be replaced by the first star’s color.\n\n"
            + "\n".join(star_examples)
            + "\n\n"
            + "**REFERRING (use 0 stars)**: include a description in the prompt.\n\n"
            + "\n".join(referring_examples)
            + "\n\n"
            + "**OBJECT (RefCOCO-style, use 0 stars)**: use an object referring expression as `{class_name}`.\n\n"
            + "\n".join(object_examples)
        )

    def _tensor_chw_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):
            pass
        elif t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected tensor shape for image: {tuple(t.shape)}")
        t = t.to(dtype=torch.float32)
        t = t.clamp(0.0, 1.0)
        hwc = (t * 255.0).byte().permute(1, 2, 0).numpy()
        if hwc.shape[2] == 1:
            hwc = np.repeat(hwc, 3, axis=2)
        return hwc

    def _png_bytes_from_uint8_hwc(img: np.ndarray) -> bytes:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected HWC uint8 RGB, got {img.shape} {img.dtype}")
        pil = Image.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def _png_bytes_from_uint8_mask(mask: np.ndarray) -> bytes:
        pil = Image.fromarray(mask.astype(np.uint8), mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def _create_cyan_overlay(base_rgb_uint8: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Deterministic overlay: blend mask region with CYAN (RGB 0,255,255).
        base_rgb_uint8: HxWx3 uint8
        mask_bool: HxW bool
        """
        base = base_rgb_uint8.astype(np.float32)
        overlay = base.copy()
        cyan = np.array([0.0, 255.0, 255.0], dtype=np.float32)
        overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + cyan * alpha
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def on_download(orig_pil, clean_tensor_chw, star_tensor_chw, pred_mask_bin):
        """
        Create a zip containing:
        - original_1024.png (resized+center-cropped, no stars)
        - with_star_1024.png (model input with stars)
        - pred_mask_1024.png (binary)
        - overlay_on_original_1024.png
        - overlay_on_star_1024.png
        """
        if clean_tensor_chw is None or star_tensor_chw is None or pred_mask_bin is None:
            return None

        orig_1024 = _tensor_chw_to_uint8_hwc(clean_tensor_chw)
        star_1024 = _tensor_chw_to_uint8_hwc(star_tensor_chw)

        mask = pred_mask_bin.astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bool = (mask > 0)
        overlay_orig = _create_cyan_overlay(orig_1024, mask_bool, alpha=0.45)
        overlay_star = _create_cyan_overlay(star_1024, mask_bool, alpha=0.45)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{ts}.zip")
        tmp.close()

        with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("original_1024.png", _png_bytes_from_uint8_hwc(orig_1024))
            zf.writestr("with_star_1024.png", _png_bytes_from_uint8_hwc(star_1024))
            zf.writestr("pred_mask_1024.png", _png_bytes_from_uint8_mask(mask))
            zf.writestr("overlay_on_original_1024.png", _png_bytes_from_uint8_hwc(overlay_orig))
            zf.writestr("overlay_on_star_1024.png", _png_bytes_from_uint8_hwc(overlay_star))
            if orig_pil is not None:
                buf = io.BytesIO()
                orig_pil.save(buf, format="PNG")
                zf.writestr("original_fullres.png", buf.getvalue())

        return tmp.name

    def on_image_upload(image_np_or_pil):
        if image_np_or_pil is None:
            return None, [], None, None, None, None, "Upload an image to start."

        if isinstance(image_np_or_pil, np.ndarray):
            orig_pil = Image.fromarray(image_np_or_pil.astype(np.uint8))
        else:
            orig_pil = image_np_or_pil

        clean_chw = demo_model.resize_image_to_square(orig_pil, 1024)  # CHW [0,1]
        disp_chw = clean_chw.clone()
        disp_np = np.array(transforms.ToPILImage()(disp_chw))  # HWC uint8

        return (
            disp_np,  # input_image (display)
            [],  # coords_state
            None,  # fixed_color_state
            orig_pil,  # orig_pil_state
            clean_chw,  # clean_tensor_state
            disp_chw,  # disp_tensor_state
            "Image loaded. Click up to 5 points, then Submit.",
        )

    def _add_star_to_tensor(disp_tensor_chw, coords_1024, fixed_color, h, w):
        marker_size = 32

        disp = disp_tensor_chw
        try:
            if fixed_color is None:
                disp, c = add_star_marker(disp, h, w, size=marker_size)
                fixed_color = c or "blue"
            else:
                disp, _ = add_star_marker(
                    disp, h, w, size=marker_size, color=fixed_color
                )
        except TypeError:
            disp, c = add_star_marker(disp, h, w, size=marker_size)
            if fixed_color is None:
                fixed_color = c or "blue"

        coords_1024 = coords_1024 + [(h, w)]
        return disp, coords_1024, fixed_color

    def on_click_add_star(
        image_disp_np, coords_1024, fixed_color, disp_tensor_chw, evt: gr.SelectData
    ):
        if image_disp_np is None or disp_tensor_chw is None:
            return (
                image_disp_np,
                coords_1024,
                fixed_color,
                disp_tensor_chw,
                "Please upload an image first.",
            )
        if evt is None:
            return (
                image_disp_np,
                coords_1024,
                fixed_color,
                disp_tensor_chw,
                "Click anywhere on the image to add a star.",
            )
        if coords_1024 is None:
            coords_1024 = []
        if len(coords_1024) >= MAX_STARS:
            return (
                image_disp_np,
                coords_1024,
                fixed_color,
                disp_tensor_chw,
                f"Max {MAX_STARS} stars reached.",
            )

        h, w = int(evt.index[1]), int(evt.index[0])
        disp, coords_1024, fixed_color = _add_star_to_tensor(
            disp_tensor_chw, coords_1024, fixed_color, h, w
        )
        disp_np = np.array(transforms.ToPILImage()(disp))
        return (
            disp_np,
            coords_1024,
            fixed_color,
            disp,
            f"Star #{len(coords_1024)} @ (h={h}, w={w}).",
        )

    def on_manual_add_star(coords_1024, fixed_color, disp_tensor_chw, h_value, w_value):
        if disp_tensor_chw is None:
            return (
                None,
                coords_1024 or [],
                fixed_color,
                None,
                "Please upload an image first.",
            )
        if h_value is None or w_value is None:
            return (
                (
                    np.array(transforms.ToPILImage()(disp_tensor_chw))
                    if disp_tensor_chw is not None
                    else None
                ),
                coords_1024 or [],
                fixed_color,
                disp_tensor_chw,
                "Provide both h and w (0-1023) before adding.",
            )
        if coords_1024 is None:
            coords_1024 = []
        if len(coords_1024) >= MAX_STARS:
            return (
                np.array(transforms.ToPILImage()(disp_tensor_chw)),
                coords_1024,
                fixed_color,
                disp_tensor_chw,
                f"Max {MAX_STARS} stars reached.",
            )

        h = int(max(0, min(1023, round(h_value))))
        w = int(max(0, min(1023, round(w_value))))
        disp, coords_1024, fixed_color = _add_star_to_tensor(
            disp_tensor_chw, coords_1024, fixed_color, h, w
        )
        disp_np = np.array(transforms.ToPILImage()(disp))
        return (
            disp_np,
            coords_1024,
            fixed_color,
            disp,
            f"Manual star #{len(coords_1024)} @ (h={h}, w={w}).",
        )

    def on_undo_last(coords_1024, fixed_color, clean_tensor_chw):
        if clean_tensor_chw is None:
            return None, coords_1024, fixed_color, None, "Nothing to undo."
        if not coords_1024:
            return (
                np.array(transforms.ToPILImage()(clean_tensor_chw.clone())),
                [],
                None,
                clean_tensor_chw.clone(),
                "Nothing to undo.",
            )

        new_coords = coords_1024[:-1]
        disp = clean_tensor_chw.clone()
        latched = fixed_color

        marker_size = max(8, int(1024 // 32))
        for i, (h, w) in enumerate(new_coords):
            try:
                if i == 0 and latched is None:
                    disp, c = add_star_marker(disp, int(h), int(w), size=marker_size)
                    latched = c or "blue"
                else:
                    disp, _ = add_star_marker(
                        disp, int(h), int(w), size=marker_size, color=latched
                    )
            except TypeError:
                disp, c = add_star_marker(disp, int(h), int(w), size=marker_size)
                if i == 0 and latched is None:
                    latched = c or "blue"

        disp_np = np.array(transforms.ToPILImage()(disp))
        return (
            disp_np,
            new_coords,
            latched,
            disp,
            f"Removed last star. {len(new_coords)} remaining.",
        )

    def on_clear_stars(clean_tensor_chw):
        if clean_tensor_chw is None:
            return None, [], None, None, "Nothing to clear."
        disp = clean_tensor_chw.clone()
        disp_np = np.array(transforms.ToPILImage()(disp))
        return disp_np, [], None, disp, "Cleared all stars."

    def on_submit(orig_pil, coords_1024, text_prompt, fixed_color, clean_tensor_chw):
        if orig_pil is None:
            return None, None, "Please upload an image first.", None, None, None

        coords_1024 = coords_1024 or []

        pred_mask, raw_mask, image_1024_with_star, latched_color, global_enc_image = (
            demo_model.inference(
                orig_pil, coords_1024, text_prompt, fixed_star_color=fixed_color
            )
        )
        if pred_mask is None:
            return None, None, f"Inference failed: {raw_mask}", None, None, None

        if len(coords_1024) == 0:
            final_prompt = text_prompt.replace("<COLOR>", "").replace("  ", " ").strip()
            status = f"REFERRING task (0 stars).\nPrompt: {final_prompt}"
        else:
            used_color = fixed_color or latched_color or "blue"
            final_prompt = text_prompt.replace("<COLOR>", used_color)
            status = f"STAR task with {len(coords_1024)} point(s). Color={used_color}.\nPrompt: {final_prompt}"

        print(f"[DEBUG] on_submit: Final prompt used: {final_prompt}")

        mask_2d = pred_mask
        if mask_2d.ndim == 3:
            mask_2d = mask_2d[0] if mask_2d.shape[0] == 1 else mask_2d[..., 0]
        mask_bool = mask_2d > 0

        if clean_tensor_chw is not None:
            orig_1024_np = _tensor_chw_to_uint8_hwc(clean_tensor_chw)
            overlay_np = _create_cyan_overlay(orig_1024_np, mask_bool)
        else:
            overlay_np = None

        binary_np = (mask_bool.astype(np.uint8) * 255)

        return overlay_np, binary_np, status, pred_mask, image_1024_with_star, final_prompt

    with gr.Blocks(title="MAOAM-GLaMM Demo") as interface:
        gr.Markdown("# MAOAM-GLaMM Demo")
        gr.Markdown(
            "1) Upload an image. &nbsp; 2) Choose a **Selection type**. &nbsp; "
            "3) Follow the hint below the prompt. &nbsp; 4) Press **Submit**."
        )

        coords_state = gr.State([])  # list[(h,w)] in 1024 space
        fixed_color_state = gr.State(None)  # str like "blue"
        orig_pil_state = gr.State(None)  # original PIL
        clean_tensor_state = gr.State(None)  # CHW clean 1024 torch.Tensor [0,1]
        disp_tensor_state = gr.State(None)  # CHW display tensor with stars
        last_pred_mask_state = gr.State(None)  # np.uint8 HxW mask (0/255)
        last_star_tensor_state = gr.State(None)  # CHW tensor with stars (1024)
        last_prompt_state = gr.State(None)  # exact prompt string after <COLOR> substitution

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input / Click to add star(s)", type="numpy", height=400
                )
                selection_dropdown = gr.Dropdown(
                    choices=list(_SELECTION_MODES.keys()),
                    value=_DEFAULT_MODE,
                    label="Selection type",
                )
                text_prompt = gr.Textbox(
                    label="Text prompt",
                    value=_SELECTION_MODES[_DEFAULT_MODE][0],
                    lines=3,
                )
                hint_md = gr.Markdown(_SELECTION_MODES[_DEFAULT_MODE][1])
                with gr.Row():
                    undo_btn = gr.Button("Undo last star", variant="secondary")
                    clear_stars_btn = gr.Button("Clear stars", variant="secondary")
                submit_btn = gr.Button("Submit", variant="primary")
                download_btn = gr.Button("Download (original+mask+overlays)", variant="secondary")
                download_file = gr.File(label="Download zip")

            with gr.Column(scale=1):
                overlay_image = gr.Image(label="Overlaid Image", height=400)
                binary_mask_image = gr.Image(label="Binary Mask", height=400)
                status_text = gr.Textbox(
                    label="Status",
                    value="Upload an image, click up to 5 star points, then Submit.",
                    interactive=False,
                )
                coords_table = gr.Dataframe(
                    headers=["h", "w"],
                    datatype=["number", "number"],
                    row_count=5,
                    col_count=(2, "fixed"),
                    interactive=False,
                    label="Star coordinates (1024 space)",
                )

        input_image.upload(
            on_image_upload,
            inputs=[input_image],
            outputs=[
                input_image,
                coords_state,
                fixed_color_state,
                orig_pil_state,
                clean_tensor_state,
                disp_tensor_state,
                status_text,
            ],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        input_image.select(
            on_click_add_star,
            inputs=[input_image, coords_state, fixed_color_state, disp_tensor_state],
            outputs=[
                input_image,
                coords_state,
                fixed_color_state,
                disp_tensor_state,
                status_text,
            ],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        undo_btn.click(
            on_undo_last,
            inputs=[coords_state, fixed_color_state, clean_tensor_state],
            outputs=[
                input_image,
                coords_state,
                fixed_color_state,
                disp_tensor_state,
                status_text,
            ],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        clear_stars_btn.click(
            on_clear_stars,
            inputs=[clean_tensor_state],
            outputs=[
                input_image,
                coords_state,
                fixed_color_state,
                disp_tensor_state,
                status_text,
            ],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        submit_btn.click(
            on_submit,
            inputs=[orig_pil_state, coords_state, text_prompt, fixed_color_state, clean_tensor_state],
            outputs=[
                overlay_image,
                binary_mask_image,
                status_text,
                last_pred_mask_state,
                last_star_tensor_state,
                last_prompt_state,
            ],
        )

        download_btn.click(
            lambda orig_pil, clean, star, mask, prompt: _download_with_prompt(
                orig_pil, clean, star, mask, prompt
            ),
            inputs=[
                orig_pil_state,
                clean_tensor_state,
                last_star_tensor_state,
                last_pred_mask_state,
                last_prompt_state,
            ],
            outputs=[download_file],
        )

        selection_dropdown.change(
            on_selection_change,
            inputs=[selection_dropdown],
            outputs=[text_prompt, hint_md],
        )

    return interface


def _download_with_prompt(orig_pil, clean_tensor_chw, star_tensor_chw, pred_mask_bin, prompt_text):
    """
    Thin wrapper around on_download that also injects the final prompt string into the zip.
    """
    if clean_tensor_chw is None or star_tensor_chw is None or pred_mask_bin is None:
        return None

    def _tensor_chw_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
        if t.ndim == 3 and t.shape[0] in (1, 3):
            pass
        elif t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        hwc = (t * 255.0).byte().permute(1, 2, 0).numpy()
        if hwc.shape[2] == 1:
            hwc = np.repeat(hwc, 3, axis=2)
        return hwc

    def _png_bytes_from_uint8_hwc(img: np.ndarray) -> bytes:
        pil = Image.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    def _png_bytes_from_uint8_mask(mask: np.ndarray) -> bytes:
        pil = Image.fromarray(mask.astype(np.uint8), mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    orig_1024 = _tensor_chw_to_uint8_hwc(clean_tensor_chw)
    star_1024 = _tensor_chw_to_uint8_hwc(star_tensor_chw)

    mask = pred_mask_bin.astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = (mask > 0)
    cyan = np.array([0.0, 255.0, 255.0], dtype=np.float32)
    base_o = orig_1024.astype(np.float32)
    base_s = star_1024.astype(np.float32)
    overlay_orig = base_o.copy()
    overlay_star = base_s.copy()
    overlay_orig[mask_bool] = overlay_orig[mask_bool] * (1.0 - 0.45) + cyan * 0.45
    overlay_star[mask_bool] = overlay_star[mask_bool] * (1.0 - 0.45) + cyan * 0.45
    overlay_orig = np.clip(overlay_orig, 0, 255).astype(np.uint8)
    overlay_star = np.clip(overlay_star, 0, 255).astype(np.uint8)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{ts}.zip")
    tmp.close()

    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("original_1024.png", _png_bytes_from_uint8_hwc(orig_1024))
        zf.writestr("with_star_1024.png", _png_bytes_from_uint8_hwc(star_1024))
        zf.writestr("pred_mask_1024.png", _png_bytes_from_uint8_mask(mask))
        zf.writestr("overlay_on_original_1024.png", _png_bytes_from_uint8_hwc(overlay_orig))
        zf.writestr("overlay_on_star_1024.png", _png_bytes_from_uint8_hwc(overlay_star))
        if orig_pil is not None:
            buf = io.BytesIO()
            orig_pil.save(buf, format="PNG")
            zf.writestr("original_fullres.png", buf.getvalue())
        if prompt_text is not None:
            zf.writestr("prompt.txt", str(prompt_text))

    return tmp.name


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="GLaMM Gradio Demo")
    parser.add_argument(
        "--model_path",
        default="MBZUAI/GLaMM-GranD-Pretrained",
        help="Path or HF ID for the base GLaMM model (defaults to pretrained GranD).",
    )
    parser.add_argument(
        "--vision_pretrained",
        default=os.environ.get("SAM_CKPT", ""),
        help="Path to SAM ViT-H weights (or set SAM_CKPT env var).",
    )
    parser.add_argument(
        "--vision_tower",
        default="openai/clip-vit-large-patch14-336",
        help="Vision tower model",
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained model"
    )
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument(
        "--share", default=False, action="store_true", help="Create public link"
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Path to mp_rank_00_model_states.pt (or similar) checkpoint to load.",
    )

    args = parser.parse_args()

    if "refseg_ep15" in args.model_path:
        model_title = "GLaMM RefSeg"
    elif "baseline_ep10" in args.model_path:
        model_title = "GLaMM Pretrained"
    else:
        model_title = "GLaMM"

    args_dict = {
        "model_max_length": 1536,
        "use_mm_start_end": True,
        "pretrained": args.pretrained,
        "train_mask_decoder": True,
        "out_dim": 256,
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "mm_vision_select_layer": -2,
        "pretrain_mm_mlp_adapter": "",
        "tune_mm_mlp_adapter": False,
        "freeze_mm_mlp_adapter": False,
        "mm_use_im_start_end": True,
        "with_region": True,
        "conv_type": "llava_v1",
        "lora_r": 0,
        "resume": args.resume,
    }

    print(f"Initializing {model_title} model...")
    demo_model = GLaMMDemo(args.model_path, args_dict)
    print("Model loaded successfully!")

    print("Creating Gradio interface...")
    interface = create_gradio_interface(demo_model, model_title)

    print(f"Launching demo on port {args.port}...")
    interface.launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, debug=False
    )


if __name__ == "__main__":
    main()
