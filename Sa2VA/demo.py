# run:
# python Sa2VA/demo.py --cfg projects/sa2va/configs/glamm_qwen25_7b_material_only.py \
#                      --resume /path/to/mp_rank_00_model_states.pt

import argparse
import os
import pickle
import random
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Path setup MUST happen before any `projects.sa2va...` import because
# projects.sa2va.datasets.__init__ transitively imports GLaMM's `dataset.*`.
MMSEG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SA2VA_ROOT = os.path.dirname(os.path.abspath(__file__))
GLAMM_ROOT = os.path.join(MMSEG_ROOT, "GLaMM")
for _p in (MMSEG_ROOT, SA2VA_ROOT, GLAMM_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.registry import BUILDER
from mmengine.config import Config

from projects.sa2va.gradio.app_utils import process_markdown, show_mask_pred
from projects.sa2va.datasets.data_utils import sa2va_collect_fn_multitask
from projects.sa2va.datasets.base import Sa2VABaseDataset
from projects.sa2va.datasets.common import ANSWER_LIST
from projects.sa2va.models.sa2va import Sa2VAModel

from utils.hm_utils import add_star_marker
from dataset.utils.utils import STAR_QUESTIONS, REFERRING_QUESTIONS, SEG_QUESTIONS, VQA_QUESTIONS, TASK_PROMPT

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def _torch_load_ckpt_safely(path: str):
    """
    PyTorch 2.6 changed torch.load default weights_only=True, which can break loading
    older / DeepSpeed checkpoints that include pickled objects (e.g. ConfigDict).
    We try weights_only=True first, then retry with weights_only=False.
    """
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
            # Older torch without weights_only kwarg
            return torch.load(path, map_location="cpu")


def _load_state_dict_from_mp_rank(path: str) -> dict:
    ckpt = _torch_load_ckpt_safely(path)
    if isinstance(ckpt, dict):
        if "module" in ckpt and isinstance(ckpt["module"], dict):
            state_dict = ckpt["module"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}: {type(state_dict)}")
    return state_dict

def parse_args(args):
    parser = argparse.ArgumentParser(description="Sa2VA Demo")
    parser.add_argument(
        "--cfg",
        default=None,
        help="MMEngine config to build the *training* Sa2VAModel (recommended for best parity). "
             "Example: projects/sa2va/configs/glamm_qwen25_7b_all.py",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to load into the MMEngine model when using --cfg. "
             "Supports DeepSpeed mp_rank_00_model_states.pt.",
    )
    parser.add_argument(
        'hf_path',
        nargs='?',
        default=None,
        help='Sa2VA HuggingFace model ID or local HF-style model directory. '
             'Optional — only used when --cfg is not provided.',
    )
    parser.add_argument(
        '--base_model', default=None,
        help='When hf_path is a .pt/.pth file, base HF model ID/directory for '
             'model+tokenizer construction.',
    )
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--tokenizer_path', default=None)
    parser.add_argument(
        '--sam2_ckpt', default=None,
        help='SAM2 grounding-encoder checkpoint path (overrides the placeholder '
             'in the training cfg). Default: $SAM2_CKPT or weights/sam2_hiera_large.pt.',
    )
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    return parser.parse_args(args)


class _InteractivePacker(Sa2VABaseDataset):
    """Minimal Sa2VABaseDataset subclass to reuse train-time text/image packing logic."""

    def real_len(self):
        return 1

    def prepare_data(self, index: int):
        raise NotImplementedError

    def process_qwen_image(self, img_chw_float: torch.Tensor, min_pixels: int, max_pixels: int):
        img_chw_float = img_chw_float.clamp(0.0, 1.0)
        pil = to_pil_image(img_chw_float)
        merge_length = self.preprocessor.image_processor.merge_size ** 2
        d = self.preprocessor.image_processor(
            images=[pil],
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
        )
        pixel_values = torch.as_tensor(d["pixel_values"], dtype=torch.float32)
        image_grid_thw = torch.as_tensor(d["image_grid_thw"], dtype=torch.long)
        num_image_tokens = int(image_grid_thw[0].prod().item()) // int(merge_length)
        return pixel_values, image_grid_thw, num_image_tokens

    def pack_task_qwen(
        self,
        qwen_image_chw_float: torch.Tensor,
        question: str,
        answer: str,
        qwen_min_pixels: int,
        qwen_max_pixels: int,
    ) -> dict:
        pixel_values, image_grid_thw, num_image_tokens = self.process_qwen_image(
            qwen_image_chw_float, min_pixels=qwen_min_pixels, max_pixels=qwen_max_pixels
        )
        image_token_str = self._create_image_token_string(num_image_tokens)
        conv = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
        conv = self._process_conversations_for_encoding(conv, image_token_str=image_token_str, is_video=False)
        conv_prompt = conv[0]["input"] if len(conv) > 0 and "input" in conv[0] else ""
        token_dict = self.get_inputid_labels(conv)
        return {
            "input_ids": token_dict["input_ids"],
            "labels": token_dict["labels"],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "convs": conv_prompt,
            "question": question,
        }

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


def _image_to_tensor(image: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
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
            image_tensor = image_tensor / 255.0 if image_tensor.max() > 1.0 else image_tensor
    else:
        image_tensor = transforms.ToTensor()(image)
    return _ensure_unit_range(image_tensor)


def _create_cyan_overlay(base_rgb_uint8: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = base_rgb_uint8.astype(np.float32)
    overlay = base.copy()
    cyan = np.array([0.0, 255.0, 255.0], dtype=np.float32)
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + cyan * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def resize_image_to_square(image: Image.Image, target_size: int = 1024) -> torch.Tensor:
    """Resize image to square target size using resize + center crop (GLaMM demo behavior)."""
    image_tensor = _image_to_tensor(image)
    h, w = image_tensor.shape[-2:]
    if h < w:
        new_h = target_size
        new_w = int(target_size * w / h)
    else:
        new_w = target_size
        new_h = int(target_size * h / w)
    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    image_tensor = transforms.CenterCrop((target_size, target_size))(image_tensor)
    return image_tensor


def _overlay_all_stars_1024(base_tensor_1024_chw: torch.Tensor, coords_1024, fixed_color: str | None):
    """
    Overlay stars on a CHW [0,1] 1024 tensor using add_star_marker ONLY.
    Returns (tensor_with_stars, latched_color).
    """
    img = base_tensor_1024_chw.clone()
    latched = fixed_color
    marker_size = max(8, int(1024 // 32))
    for i, (hh, ww) in enumerate(coords_1024):
        h = int(max(0, min(1023, hh)))
        w = int(max(0, min(1023, ww)))
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


def _run_model(image_pil: Image.Image, text: str):
    # NOTE: This gradio demo is intentionally single-turn.
    # Keep a stable "model sees" string for debugging.
    model_text = text if "<image>" in text else "<image>" + text
    input_dict = {
        "image": image_pil,
        "text": model_text,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
        "processor": processor,
    }
    return sa2va_model.predict_forward(**input_dict), model_text

def init_models(args):
    # If a training config is provided, build the same MMEngine `Sa2VAModel` used in training
    # and reuse the dataset packer logic to create the exact `tasks` batch.
    if args.cfg is not None:
        if args.resume is None:
            raise ValueError("When using --cfg, you must also provide --resume (mp_rank_00_model_states.pt or .pth).")

        cfg = Config.fromfile(args.cfg)
        # The demo loads the requested checkpoint itself, so only resolve the
        # SAM2 dependency from the CLI/environment before building the model.
        if isinstance(cfg.model, dict):
            cfg.model.pop("pretrained_pth", None)
            sam2_ckpt = args.sam2_ckpt or os.environ.get("SAM2_CKPT") \
                        or os.path.join(MMSEG_ROOT, "weights", "sam2_hiera_large.pt")
            if not os.path.exists(sam2_ckpt):
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found at {sam2_ckpt}. "
                    f"Pass --sam2_ckpt or set SAM2_CKPT env var."
                )
            if isinstance(cfg.model.get("grounding_encoder"), dict):
                cfg.model["grounding_encoder"]["ckpt_path"] = os.path.abspath(sam2_ckpt)
        model = BUILDER.build(cfg.model)

        # Load checkpoint (supports DeepSpeed mp_rank shards)
        resume_path = str(args.resume)
        # Avoid torch.load weights_only pitfalls by using a safe loader here.
        state_dict = _load_state_dict_from_mp_rank(resume_path)
        # Strip common DDP prefix
        if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint from {resume_path}")
        print(f"[INFO] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        model = model.eval().cuda()

        # Build a packer that matches train-time encoding (prompt_template, tokenizer, preprocessor, etc.)
        # We mirror the config values used in `glamm_qwen25_7b_all.py`.
        prompt_template = getattr(cfg, "prompt_template", PROMPT_TEMPLATE.qwen_chat)
        max_length = int(getattr(cfg, "max_length", 8192))
        special_tokens = list(getattr(cfg, "special_tokens", ["[SEG]"]))
        tokenizer_cfg = getattr(cfg, "tokenizer", None)
        # In the configs, preprocessor lives under sa2va_glamm_default_dataset_configs.preprocessor
        preprocessor_cfg = None
        if hasattr(cfg, "sa2va_glamm_default_dataset_configs"):
            preprocessor_cfg = cfg.sa2va_glamm_default_dataset_configs.get("preprocessor", None)

        packer = _InteractivePacker(
            tokenizer=tokenizer_cfg,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type="qwen",
            preprocessor=preprocessor_cfg,
            repeats=1.0,
            name="Sa2VAInteractivePacker",
        )

        # Match dataset defaults used in Sa2VAMaterialDataset
        qwen_min_pixels = int(getattr(cfg, "qwen_min_pixels", 512 * 28 * 28))
        qwen_max_pixels = int(getattr(cfg, "qwen_max_pixels", 2048 * 28 * 28))

        return ("mmengine", model, packer, qwen_min_pixels, qwen_max_pixels)

    if args.hf_path is None:
        raise ValueError(
            "When --cfg is not provided, you must pass a positional HF model path "
            "(or HF model ID) so the model can be loaded via AutoModel.from_pretrained."
        )
    model_path = args.hf_path
    ckpt_path = args.ckpt

    # Convenience: allow passing a checkpoint as the positional arg.
    if os.path.isfile(model_path) and model_path.lower().endswith(('.pt', '.pth')):
        if args.base_model is None:
            raise ValueError(
                "You passed a .pt/.pth file as hf_path, but did not provide --base_model. "
                "AutoModel.from_pretrained still needs a HuggingFace model ID or local model directory "
                "(config.json + tokenizer files) to construct the model."
            )
        ckpt_path = model_path
        model_path = args.base_model

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or model_path,
        trust_remote_code=True,
    )
    # Some Sa2VA variants (e.g. Qwen2.5-VL) require a Processor in predict_forward().
    try:
        processor = AutoProcessor.from_pretrained(
            args.tokenizer_path or model_path,
            trust_remote_code=True,
        )
    except Exception as e:
        processor = None
        print(f"Info: AutoProcessor.from_pretrained failed ({type(e).__name__}): {e}. Continuing without processor.")

    if ckpt_path is not None:
        # PyTorch 2.6 changed torch.load default weights_only=True, which can break
        # loading older / DeepSpeed checkpoints that rely on pickling.
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except (pickle.UnpicklingError, TypeError) as e:
            # - pickle.UnpicklingError: not compatible with weights_only=True
            # - TypeError: older torch without weights_only kwarg
            print(
                f"Warning: torch.load(weights_only=True) failed for {ckpt_path}: {e}\n"
                "Retrying with weights_only=False. Only do this if the checkpoint is from a trusted source."
            )
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # common checkpoint formats
        if isinstance(ckpt, dict):
            # DeepSpeed/ZeRO commonly stores model weights under "module"
            if 'module' in ckpt and isinstance(ckpt['module'], dict):
                ckpt = ckpt['module']
            elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                ckpt = ckpt['state_dict']
            elif 'model' in ckpt and isinstance(ckpt['model'], dict):
                ckpt = ckpt['model']
            elif 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
                ckpt = ckpt['model_state_dict']

        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Unsupported checkpoint format at {ckpt_path}. Expected a state_dict dict or a dict "
                f"containing one of: state_dict/model/model_state_dict, but got: {type(ckpt)}"
            )

        # strip common DDP prefix
        if len(ckpt) > 0 and all(k.startswith('module.') for k in ckpt.keys()):
            ckpt = {k[len('module.'):]: v for k, v in ckpt.items()}

        # Load only matching keys to avoid huge "unexpected keys" spam when the checkpoint is
        # partial (e.g. LoRA-only) or from a different wrapper.
        model_sd = model.state_dict()
        filtered = {}
        for k, v in ckpt.items():
            if k not in model_sd:
                continue
            try:
                if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape != model_sd[k].shape:
                    continue
            except Exception:
                continue
            filtered[k] = v

        incompatible = model.load_state_dict(filtered, strict=False)
        missing_keys = getattr(incompatible, "missing_keys", incompatible[0])
        unexpected_keys = getattr(incompatible, "unexpected_keys", incompatible[1])

        print(f'Load checkpoint from {ckpt_path}')
        print(f'Info: checkpoint tensors={len(ckpt)}; loaded (shape-matched) tensors={len(filtered)}')
        # Many training checkpoints are LoRA-only / partial weights; missing keys are expected.
        if missing_keys:
            if len(filtered) < 2000:
                print(
                    f'Info: loaded a partial checkpoint ({len(filtered)} tensors). '
                    f'Missing keys are expected (missing={len(missing_keys)}). '
                    f'Example: {missing_keys[:5]}'
                )
            else:
                print(f'Warning: missing keys ({len(missing_keys)}). Example: {missing_keys[:5]}')
        if unexpected_keys:
            print(f'Warning: unexpected keys ({len(unexpected_keys)}). Example: {unexpected_keys[:5]}')
    return model, tokenizer, processor

class global_infos:
    image_width = 0
    image_height = 0

    image_for_show = None
    image = None

if __name__ == "__main__":
    # get parse args and set models
    args = parse_args(sys.argv[1:])

    _init = init_models(args)
    if isinstance(_init, tuple) and len(_init) >= 2 and _init[0] == "mmengine":
        MODEL_MODE = "mmengine"
        sa2va_model = _init[1]
        packer = _init[2]
        qwen_min_pixels = int(_init[3])
        qwen_max_pixels = int(_init[4])
        tokenizer = None
        processor = None
        print(f"[INFO] Running in MODEL_MODE=mmengine (train-parity path) using cfg={args.cfg}")
    else:
        MODEL_MODE = "hf"
        sa2va_model, tokenizer, processor = _init
        packer = None
        qwen_min_pixels = None
        qwen_max_pixels = None
        print("[INFO] Running in MODEL_MODE=hf (predict_forward generation path)")

    MAX_STARS = 5

    def on_image_upload(image_np_or_pil):
        if image_np_or_pil is None:
            return None, [], None, None, None, "Upload an image to start."

        if isinstance(image_np_or_pil, np.ndarray):
            orig_pil = Image.fromarray(image_np_or_pil.astype(np.uint8)).convert("RGB")
        else:
            orig_pil = image_np_or_pil.convert("RGB")

        clean_chw = resize_image_to_square(orig_pil, 1024)  # CHW [0,1]
        disp_chw = clean_chw.clone()
        disp_np = np.array(transforms.ToPILImage()(disp_chw))  # HWC uint8

        clean_pil_1024 = transforms.ToPILImage()(clean_chw)
        return disp_np, [], None, orig_pil, clean_pil_1024, clean_chw, disp_chw, "Image loaded. Click up to 5 points, then Submit."

    def _add_star_to_tensor(disp_tensor_chw, coords_1024, fixed_color, h, w):
        marker_size = 32
        disp = disp_tensor_chw
        try:
            if fixed_color is None:
                disp, c = add_star_marker(disp, h, w, size=marker_size)
                fixed_color = c or "blue"
            else:
                disp, _ = add_star_marker(disp, h, w, size=marker_size, color=fixed_color)
        except TypeError:
            disp, c = add_star_marker(disp, h, w, size=marker_size)
            if fixed_color is None:
                fixed_color = c or "blue"

        coords_1024 = (coords_1024 or []) + [(h, w)]
        return disp, coords_1024, fixed_color

    def on_click_add_star(image_disp_np, coords_1024, fixed_color, disp_tensor_chw, evt: gr.SelectData):
        if image_disp_np is None or disp_tensor_chw is None:
            return image_disp_np, coords_1024, fixed_color, disp_tensor_chw, "Please upload an image first."
        if evt is None:
            return image_disp_np, coords_1024, fixed_color, disp_tensor_chw, "Click anywhere on the image to add a star."
        if coords_1024 is None:
            coords_1024 = []
        if len(coords_1024) >= MAX_STARS:
            return image_disp_np, coords_1024, fixed_color, disp_tensor_chw, f"Max {MAX_STARS} stars reached."

        h, w = int(evt.index[1]), int(evt.index[0])
        disp, coords_1024, fixed_color = _add_star_to_tensor(disp_tensor_chw, coords_1024, fixed_color, h, w)
        disp_np = np.array(transforms.ToPILImage()(disp))
        return disp_np, coords_1024, fixed_color, disp, f"Star #{len(coords_1024)} @ (h={h}, w={w})."

    def on_undo_last(coords_1024, fixed_color, clean_tensor_chw):
        if clean_tensor_chw is None:
            return None, coords_1024, fixed_color, None, "Nothing to undo."
        if not coords_1024:
            disp = clean_tensor_chw.clone()
            return np.array(transforms.ToPILImage()(disp)), [], None, disp, "Nothing to undo."

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
                    disp, _ = add_star_marker(disp, int(h), int(w), size=marker_size, color=latched)
            except TypeError:
                disp, c = add_star_marker(disp, int(h), int(w), size=marker_size)
                if i == 0 and latched is None:
                    latched = c or "blue"

        return np.array(transforms.ToPILImage()(disp)), new_coords, latched, disp, f"Removed last star. {len(new_coords)} remaining."

    def on_clear_stars(clean_tensor_chw):
        if clean_tensor_chw is None:
            return None, [], None, None, "Nothing to clear."
        disp = clean_tensor_chw.clone()
        return np.array(transforms.ToPILImage()(disp)), [], None, disp, "Cleared all stars."

    def on_submit(
        orig_pil_1024,
        coords_1024,
        text_prompt,
        fixed_color,
        disp_tensor_chw,
    ):
        if orig_pil_1024 is None:
            return None, None, "Please upload an image first."

        coords_1024 = coords_1024 or []
        used_color = fixed_color

        if len(coords_1024) == 0:
            final_prompt = text_prompt.replace("<COLOR>", "").replace("  ", " ").strip()
            model_image = orig_pil_1024
            status = f"REFERRING task (0 stars).\nPrompt: {final_prompt}"
        else:
            if disp_tensor_chw is None:
                base_chw = _image_to_tensor(orig_pil_1024)
            else:
                base_chw = disp_tensor_chw
            if used_color is None:
                _, used_color = _overlay_all_stars_1024(_image_to_tensor(orig_pil_1024), coords_1024, None)
            final_prompt = text_prompt.replace("<COLOR>", used_color or "blue")
            model_image = transforms.ToPILImage()(base_chw)
            status = (
                f"STAR task with {len(coords_1024)} point(s). Color={used_color or 'blue'}.\n"
                f"Prompt: {final_prompt}"
            )

        # Print what the model actually sees (GLaMM-style debug log)
        print(f"[DEBUG] on_submit: Final prompt (pre-<image>) used: {final_prompt}")

        global_infos.image_for_show = model_image
        global_infos.image = model_image

        if MODEL_MODE == "mmengine":
            # Train-parity path: build multitask batch like training (sa2va_collect_fn_multitask + Sa2VAModel._forward_multitask)
            clean_chw = _image_to_tensor(orig_pil_1024)
            g_u8 = (clean_chw.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

            # Dummy GT mask (only used to set output resolution in inference path)
            dummy_mask = torch.zeros((1, 1024, 1024), dtype=torch.uint8)

            tasks = {}
            # IMPORTANT: questions in training are normalized to "<image>\n{body}"
            question_for_model = "<image>\n" + final_prompt.strip()
            answer = random.choice(ANSWER_LIST)

            if len(coords_1024) == 0:
                # Referring uses image_without_star
                tasks["referring"] = packer.pack_task_qwen(
                    qwen_image_chw_float=clean_chw,
                    question=question_for_model,
                    answer=answer,
                    qwen_min_pixels=qwen_min_pixels,
                    qwen_max_pixels=qwen_max_pixels,
                )
                tdbg = tasks["referring"]
                instance = {
                    "src": "gradio",
                    "images_without_star": clean_chw,
                    "g_pixel_values": g_u8,
                    "masks": dummy_mask,
                    "tasks": tasks,
                }
                task_key = "referring"
            else:
                # Star uses image_star
                star_chw = _image_to_tensor(model_image)
                tasks["star"] = packer.pack_task_qwen(
                    qwen_image_chw_float=star_chw,
                    question=question_for_model,
                    answer=answer,
                    qwen_min_pixels=qwen_min_pixels,
                    qwen_max_pixels=qwen_max_pixels,
                )
                tdbg = tasks["star"]
                instance = {
                    "src": "gradio",
                    "images_star": star_chw,
                    "g_pixel_values": g_u8,
                    "masks": dummy_mask,
                    "tasks": tasks,
                }
                task_key = "star"

            try:
                seg_id = int(sa2va_model.seg_token_idx)
                seg_cnt = sum(1 for _id in tdbg["input_ids"] if int(_id) == seg_id)
            except Exception:
                seg_cnt = "unknown"
            print(
                f"[DEBUG] mmengine pack: task={task_key} "
                f"input_ids={len(tdbg['input_ids'])} labels={len(tdbg['labels'])} "
                f"seg_cnt={seg_cnt} "
                f"pixel_values={tuple(tdbg['pixel_values'].shape)} image_grid_thw={tuple(tdbg['image_grid_thw'].shape)}"
            )

            batch = sa2va_collect_fn_multitask([instance])["data"]
            batch["inference"] = True
            # Move tensors to GPU (task dict tensors are handled inside model; g_pixel_values/masks are handled in model)
            out = sa2va_model(batch, None, mode="loss")
            task_out = out.get(task_key, {})
            pred_masks = task_out.get("pred_masks", [])

            # Convert to numpy for show_mask_pred()
            pred_masks_np = []
            for m in pred_masks:
                if torch.is_tensor(m):
                    pred_masks_np.append(m.detach().cpu().numpy())
                else:
                    pred_masks_np.append(np.asarray(m))

            def _masks_to_overlay_and_binary(masks_np, orig_pil):
                orig_np = np.array(orig_pil.convert("RGB")) if orig_pil else None
                if not masks_np or orig_np is None:
                    return orig_np, None
                m = masks_np[0]
                if m.ndim == 3:
                    m = m[0]
                mb = (m > 0.5) if m.dtype != np.uint8 else (m > 0)
                if mb.shape != orig_np.shape[:2]:
                    from PIL import Image as _PI
                    mb = np.array(_PI.fromarray((mb * 255).astype(np.uint8)).resize(
                        (orig_np.shape[1], orig_np.shape[0]), _PI.NEAREST)) > 0
                return _create_cyan_overlay(orig_np, mb), (mb.astype(np.uint8) * 255)

            overlay_np, binary_np = _masks_to_overlay_and_binary(pred_masks_np, orig_pil_1024)
            status = f"{status}\nTask: {task_key} | Mode: mmengine"
            return overlay_np, binary_np, status

        # Fallback HF predict_forward generation path
        return_dict, model_text = _run_model(model_image, final_prompt)
        print(f"[DEBUG] on_submit: Model text used: {model_text}")

        hf_masks = return_dict.get("prediction_masks") or []
        hf_masks_np = [np.asarray(m) for m in hf_masks]

        def _masks_to_overlay_and_binary_hf(masks_np, orig_pil):
            orig_np = np.array(orig_pil.convert("RGB")) if orig_pil else None
            if not masks_np or orig_np is None:
                return orig_np, None
            m = masks_np[0]
            if m.ndim == 3:
                m = m[0]
            mb = (m > 0.5) if m.dtype != np.uint8 else (m > 0)
            if mb.shape != orig_np.shape[:2]:
                from PIL import Image as _PI
                mb = np.array(_PI.fromarray((mb * 255).astype(np.uint8)).resize(
                    (orig_np.shape[1], orig_np.shape[0]), _PI.NEAREST)) > 0
            return _create_cyan_overlay(orig_np, mb), (mb.astype(np.uint8) * 255)

        overlay_np, binary_np = _masks_to_overlay_and_binary_hf(hf_masks_np, orig_pil_1024)
        predict = process_markdown(return_dict.get("prediction", "").strip(), [])
        combined_status = f"{status}\n{predict}"
        return overlay_np, binary_np, combined_status

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

    with gr.Blocks(title="MAOAM-Sa2VA Demo") as demo:
        gr.Markdown("# MAOAM-Sa2VA Demo")
        gr.Markdown(
            "1) Upload an image. &nbsp; 2) Choose a **Selection type**. &nbsp; "
            "3) Follow the hint below the prompt. &nbsp; 4) Press **Submit**."
        )

        # States (mirrors GLaMM multitask demo logic)
        coords_state = gr.State([])           # list[(h,w)] in 1024 space
        fixed_color_state = gr.State(None)   # latched first star color
        fullres_pil_state = gr.State(None)   # original full-res PIL (pre-resize)
        orig_pil_state = gr.State(None)      # clean PIL 1024-square (upload-normalized)
        clean_tensor_state = gr.State(None)  # CHW clean 1024 torch.Tensor [0,1]
        disp_tensor_state = gr.State(None)   # CHW display tensor with stars

        import io as _io
        import zipfile as _zipfile
        import tempfile as _tempfile
        from datetime import datetime as _datetime

        last_overlay_state = gr.State(None)
        last_binary_state = gr.State(None)
        last_prompt_state = gr.State(None)
        last_clean_tensor_state = gr.State(None)
        last_star_tensor_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input / Click to add star(s)", type="numpy", height=400)
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
                    clear_btn = gr.Button("Clear stars", variant="secondary")
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
            outputs=[input_image, coords_state, fixed_color_state, fullres_pil_state, orig_pil_state, clean_tensor_state, disp_tensor_state, status_text],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        input_image.select(
            on_click_add_star,
            inputs=[input_image, coords_state, fixed_color_state, disp_tensor_state],
            outputs=[input_image, coords_state, fixed_color_state, disp_tensor_state, status_text],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        undo_btn.click(
            on_undo_last,
            inputs=[coords_state, fixed_color_state, clean_tensor_state],
            outputs=[input_image, coords_state, fixed_color_state, disp_tensor_state, status_text],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        clear_btn.click(
            on_clear_stars,
            inputs=[clean_tensor_state],
            outputs=[input_image, coords_state, fixed_color_state, disp_tensor_state, status_text],
        ).then(
            lambda coords: [[h, w] for (h, w) in (coords or [])],
            inputs=[coords_state],
            outputs=[coords_table],
        )

        def _sa2va_download(fullres_pil, clean_tensor_chw, star_tensor_chw, pred_mask_bin, prompt_text):
            if clean_tensor_chw is None or star_tensor_chw is None or pred_mask_bin is None:
                return None

            def _tensor_to_u8(t):
                t = t.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
                hwc = (t * 255.0).byte().permute(1, 2, 0).numpy()
                if hwc.shape[2] == 1:
                    hwc = np.repeat(hwc, 3, axis=2)
                return hwc

            def _png(arr):
                buf = _io.BytesIO()
                Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
                return buf.getvalue()

            def _png_mask(m):
                buf = _io.BytesIO()
                Image.fromarray(m.astype(np.uint8), mode="L").save(buf, format="PNG")
                return buf.getvalue()

            orig_1024 = _tensor_to_u8(clean_tensor_chw)
            star_1024 = _tensor_to_u8(star_tensor_chw)

            mask = pred_mask_bin.astype(np.uint8)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask_bool = (mask > 0)
            cyan = np.array([0.0, 255.0, 255.0], dtype=np.float32)
            overlay_orig = orig_1024.astype(np.float32)
            overlay_star = star_1024.astype(np.float32)
            overlay_orig[mask_bool] = overlay_orig[mask_bool] * 0.55 + cyan * 0.45
            overlay_star[mask_bool] = overlay_star[mask_bool] * 0.55 + cyan * 0.45
            overlay_orig = np.clip(overlay_orig, 0, 255).astype(np.uint8)
            overlay_star = np.clip(overlay_star, 0, 255).astype(np.uint8)

            ts = _datetime.now().strftime("%Y%m%d_%H%M%S")
            tmp = _tempfile.NamedTemporaryFile(delete=False, suffix=f"_{ts}.zip")
            tmp.close()
            with _zipfile.ZipFile(tmp.name, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("original_1024.png", _png(orig_1024))
                zf.writestr("with_star_1024.png", _png(star_1024))
                zf.writestr("pred_mask_1024.png", _png_mask(mask))
                zf.writestr("overlay_on_original_1024.png", _png(overlay_orig))
                zf.writestr("overlay_on_star_1024.png", _png(overlay_star))
                if fullres_pil is not None:
                    buf = _io.BytesIO()
                    fullres_pil.save(buf, format="PNG")
                    zf.writestr("original_fullres.png", buf.getvalue())
                if prompt_text:
                    zf.writestr("prompt.txt", str(prompt_text))
            return tmp.name

        submit_btn.click(
            on_submit,
            inputs=[
                orig_pil_state,
                coords_state,
                text_prompt,
                fixed_color_state,
                disp_tensor_state,
            ],
            outputs=[overlay_image, binary_mask_image, status_text],
        ).then(
            lambda ov, bm, st, clean, star: (ov, bm, st, clean, star),
            inputs=[overlay_image, binary_mask_image, status_text, clean_tensor_state, disp_tensor_state],
            outputs=[last_overlay_state, last_binary_state, last_prompt_state, last_clean_tensor_state, last_star_tensor_state],
        )

        download_btn.click(
            _sa2va_download,
            inputs=[fullres_pil_state, last_clean_tensor_state, last_star_tensor_state, last_binary_state, last_prompt_state],
            outputs=[download_file],
        )

        selection_dropdown.change(
            on_selection_change,
            inputs=[selection_dropdown],
            outputs=[text_prompt, hint_md],
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
