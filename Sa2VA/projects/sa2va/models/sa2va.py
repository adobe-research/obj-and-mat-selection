from typing import Any, Dict, Literal
from collections import OrderedDict
from pycocotools import mask as _mask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import IGNORE_INDEX

from third_parts.mmdet.models.utils.point_sample import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor

class Sa2VAModel(BaseModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 fix_number: int = 5,
                 loss_sample_points=False,
                 num_points=12544,
                 template=None,
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 training_bs:int=0,
                 weight_star: float = 0.4,
                 weight_referring: float = 0.4,
                 weight_vqa: float = 0.2,
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(tokenizer, special_tokens)

        if arch_type == 'qwen':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        if self.arch_type == 'qwen' and self.mllm.model.config.tie_word_embeddings:
            print("Untying embed_tokens and lm_head weights for Qwen model.")
            self.mllm.model.config.tie_word_embeddings = False
            lm_head = self.mllm.model.get_output_embeddings()
            if lm_head is not None:
                input_embeddings = self.mllm.model.get_input_embeddings()
                lm_head.weight = nn.Parameter(input_embeddings.weight.clone())

        in_dim = self.mllm.get_embedding_size()
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

            if self.arch_type == 'qwen':
                print("Force updating lm_head weight from pretrained state_dict.")
                lm_head_key = 'mllm.model.lm_head.weight'
                if lm_head_key in pretrained_state_dict:
                    lm_head_weight = pretrained_state_dict[lm_head_key]
                    self.mllm.model.get_output_embeddings().weight.data.copy_(lm_head_weight)
                    print(f"Successfully updated lm_head weight from key: {lm_head_key}")
                else:
                    print(f"Warning: lm_head weight key '{lm_head_key}' not found in pretrained_state_dict.")

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75
        self.fix_number = int(fix_number)

        self.template = template
        self.bs = training_bs
        self.weight_star = float(weight_star)
        self.weight_referring = float(weight_referring)
        self.weight_vqa = float(weight_vqa)

        if self.mllm.use_llm_lora:
            self.mllm.manual_prepare_llm_for_lora()

        print("\n" + "="*80)
        print("GRADIENT STATUS OF MLLM.MODEL WEIGHTS")
        print("="*80)
        
        try:
            base_model = self.mllm.model
            total_params = 0
            trainable_params = 0
            
            for name, param in base_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    grad_status = "✓ TRAINABLE"
                else:
                    grad_status = "✗ FROZEN"
                
                print(f"{name:<60} | {grad_status} | Shape: {tuple(param.shape)} | Params: {param.numel():,}")
            
            print("-" * 80)
            print(f"SUMMARY:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
            print("=" * 80)
            
        except Exception as e:
            print(f"Failed to access self.mllm.model: {e}")
            print("Available attributes in self.mllm.model:")
            print([attr for attr in dir(self.mllm.model) if not attr.startswith('_')])


    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] # required to make add_special_tokens to be False to avoid <bos> or <eos>

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def _merge_lora(self):
        if isinstance(self.mllm.model, PeftModelForCausalLM):
            self.mllm.model = self.mllm.model.merge_and_unload()
            return
        
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        state_dict_sam2 = self.grounding_encoder.state_dict(*args, prefix=prefix + 'grounding_encoder.', **kwargs)
        state_dict_text = self.text_hidden_fcs.state_dict(*args, prefix=prefix + 'text_hidden_fcs.', **kwargs)
        to_return = OrderedDict()
        to_return.update(state_dict_mllm)
        to_return.update(
            {k: v
             for k, v in state_dict_sam2.items() if k.startswith('grounding_encoder.sam2_model.sam_mask_decoder')})
        to_return.update(state_dict_text)
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=None):
        if fix_number is None:
            fix_number = self.fix_number
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((self.fix_number, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        if isinstance(data, dict) and "tasks" in data:
            return self._forward_multitask(data)
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video = self.generate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video, fix_number=self.fix_number
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)


        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        return loss_dict

    def _ce_loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )

    def _select_feats(self, feats: Dict[str, Any], idx: torch.Tensor) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in feats.items():
            if isinstance(v, list):
                out[k] = [t.index_select(0, idx) for t in v]
            elif torch.is_tensor(v):
                out[k] = v.index_select(0, idx)
            else:
                out[k] = v
        return out

    def _forward_multitask(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tasks = data["tasks"]
        g_pixel_values = data.get("g_pixel_values", None)
        gt_masks = data.get("masks", None)
        frames_per_batch = data.get("frames_per_batch", None)
        inference = bool(data.get("inference", False))

        if g_pixel_values is None or gt_masks is None or frames_per_batch is None:
            raise ValueError("Expected g_pixel_values, masks, frames_per_batch in multitask batch")

        device = next(self.parameters()).device

        g_pixel_values = torch.stack(
            [self.grounding_encoder.preprocess_image(px.to(device=device)) for px in g_pixel_values]
        ).to(dtype=self.torch_dtype, device=device)
        feats_all = self.grounding_encoder.get_sam2_feats(g_pixel_values)

        task_order = ["star", "referring", "vqa"]
        task_slices: Dict[str, slice] = {}

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        base_indices_list = []

        b_total = 0
        max_len = 0
        for tn in task_order:
            if tn not in tasks:
                continue
            td = tasks[tn]
            ti = td["input_ids"]
            ta = td["attention_mask"]
            tl = td["labels"]
            if ti.ndim != 2:
                raise ValueError(f"Expected input_ids [B,L] for task={tn}, got {tuple(ti.shape)}")
            if tl.ndim != 2:
                raise ValueError(f"Expected labels [B,L] for task={tn}, got {tuple(tl.shape)}")
            if ta.ndim != 2:
                raise ValueError(f"Expected attention_mask [B,L] for task={tn}, got {tuple(ta.shape)}")

            b = int(ti.shape[0])
            task_slices[tn] = slice(b_total, b_total + b)
            b_total += b
            max_len = max(max_len, int(ti.shape[1]))

            input_ids_list.append(ti)
            attention_mask_list.append(ta)
            labels_list.append(tl)
            pixel_values_list += list(td["pixel_values"])
            image_grid_thw_list += list(td["image_grid_thw"])
            base_indices_list.append(td["base_indices"])

        if b_total == 0:
            raise ValueError("No tasks found in multitask batch")

        def _pad_right(x: torch.Tensor, pad_value: int, tgt_len: int) -> torch.Tensor:
            if x.shape[1] == tgt_len:
                return x
            if x.shape[1] > tgt_len:
                raise ValueError("Unexpected: sequence longer than max_len")
            return F.pad(x, (0, tgt_len - x.shape[1]), value=pad_value)

        input_ids = torch.cat([_pad_right(x, 0, max_len) for x in input_ids_list], dim=0).to(device=device)
        attention_mask = torch.cat([_pad_right(x.to(torch.bool), 0, max_len) for x in attention_mask_list], dim=0).to(device=device)
        labels = torch.cat([_pad_right(x, IGNORE_INDEX, max_len) for x in labels_list], dim=0).to(device=device)
        base_indices = torch.cat([b.to(torch.long) for b in base_indices_list], dim=0).to(device=device)

        out = self.mllm(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": [pv.to(device=device) for pv in pixel_values_list],
                "image_grid_thw": [g.to(device=device) for g in image_grid_thw_list],
            },
            data_samples=None,
            mode="loss",
        )

        hidden_all = self.text_hidden_fcs(out.hidden_states[-1])
        logits_all = out.logits

        total_loss = torch.tensor(0.0, device=device)
        loss_dict: Dict[str, torch.Tensor] = {}
        if inference:
            out_dict: Dict[str, Any] = {}

        for tn, w in [("star", self.weight_star), ("referring", self.weight_referring), ("vqa", self.weight_vqa)]:
            if tn not in task_slices:
                continue
            s = task_slices[tn]
            ce = self._ce_loss_from_logits(logits_all[s], labels[s])

            if tn == "vqa":
                loss_dict[f"loss_{tn}"] = ce
                total_loss = total_loss + (w * ce)
                if inference:
                    out_dict[tn] = {
                        "loss": ce.detach(),
                        "ce_loss": ce.detach(),
                    }
                continue

            td_input_ids = input_ids[s]
            td_hidden = hidden_all[s]
            base_idx = base_indices[s]

            seg_token_mask = td_input_ids == self.seg_token_idx
            seg_token_counts = seg_token_mask.int().sum(-1)
            if int((seg_token_counts == 0).any().item()):
                raise RuntimeError(f"Found sample with 0 [SEG] tokens in task={tn}")

            pred_embeddings = td_hidden[seg_token_mask]
            pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
            if len(pred_embeddings_list_) != int(td_input_ids.shape[0]):
                raise RuntimeError("Unexpected split length mismatch for pred embeddings")

            pred_embeddings_list = [p for p in pred_embeddings_list_ if p.numel() > 0]
            if len(pred_embeddings_list) != int(td_input_ids.shape[0]):
                raise RuntimeError("Unexpected: empty pred embedding list item")

            frames_sel = [frames_per_batch[i] for i in base_idx.tolist()]
            pred_embeddings_list_video = self.generate_video_pred_embeddings(pred_embeddings_list, frames_sel)

            feats = self._select_feats(feats_all, base_idx)
            gt_masks_batch = [gt_masks[i].to(device=device) for i in base_idx.tolist()]
            gt_masks_video = self.process_video_gt_masks(gt_masks_batch, frames_sel)
            pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
                pred_embeddings_list_video, gt_masks_video, fix_number=self.fix_number
            )

            num_objs = pred_embeddings_list_video[0].shape[0]
            num_frames = len(pred_embeddings_list_video)
            language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]

            sam_states = self.grounding_encoder.get_sam2_states_from_feats(feats, expand_size=num_objs)
            pred_masks = self.grounding_encoder.inject_language_embd(
                sam_states, language_embeddings, nf_nobj=(num_frames, num_objs)
            )

            pred_masks_cat = pred_masks.flatten(0, 1)
            gt_masks_cat = torch.cat(gt_masks_video, dim=0)
            target_hw = gt_masks_cat.shape[-2:]
            pred_masks_cat = F.interpolate(
                pred_masks_cat.unsqueeze(1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            if gt_masks_cat.device != pred_masks_cat.device:
                raise RuntimeError(f"Mask device mismatch: gt={gt_masks_cat.device}, pred={pred_masks_cat.device}")

            if self.loss_sample_points:
                sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks_cat, gt_masks_cat)
                sam_loss_dice = self.loss_dice(sampled_pred_mask, sampled_gt_mask, avg_factor=(len(gt_masks_cat) + 1e-4))
                sam_loss_mask = self.loss_mask(
                    sampled_pred_mask.reshape(-1),
                    sampled_gt_mask.reshape(-1),
                    avg_factor=(pred_masks_cat.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
            else:
                sam_loss_mask = self.loss_mask(pred_masks_cat, gt_masks_cat)
                sam_loss_dice = self.loss_dice(pred_masks_cat, gt_masks_cat)

            loss_dict[f"loss_mask_{tn}"] = sam_loss_mask
            loss_dict[f"loss_dice_{tn}"] = sam_loss_dice
            loss_dict[f"loss_ce_{tn}"] = ce

            task_loss = ce + sam_loss_mask + sam_loss_dice
            loss_dict[f"loss_{tn}"] = task_loss
            total_loss = total_loss + (w * task_loss)

            if inference:
                pred_masks_up = pred_masks_cat.view(num_frames, num_objs, *target_hw)
                pred_list = []
                gt_list = []
                for i in range(int(pred_masks.shape[0])):
                    pm = pred_masks_up[i]
                    gm = gt_masks_video[i]
                    if pm.ndim != 3:
                        raise RuntimeError(f"Unexpected pred_masks[{i}] shape: {tuple(pm.shape)}")
                    if gm.ndim != 3:
                        raise RuntimeError(f"Unexpected gt_masks_video[{i}] shape: {tuple(gm.shape)}")
                    pred_list.append(pm[:1].detach())
                    gt_list.append(gm[:1].detach())

                convs = tasks[tn].get("convs", None)
                if convs is None:
                    convs = [""] * int(pred_masks.shape[0])
                out_dict[tn] = {
                    "loss": task_loss.detach(),
                    "ce_loss": ce.detach(),
                    "mask_bce_loss": sam_loss_mask.detach(),
                    "mask_dice_loss": sam_loss_dice.detach(),
                    "mask_loss": (sam_loss_mask + sam_loss_dice).detach(),
                    "pred_masks": pred_list,
                    "gt_masks": gt_list,
                    "convs": convs,
                }

        loss_dict["loss"] = total_loss
        if inference:
            out_dict["loss"] = total_loss.detach()
            return out_dict
        return loss_dict


    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def generate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        assert len(pred_embeddings_list) == len(frames_per_batch)
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        raise NotImplementedError("Sa2VA does not support preparing for generation, please use predict_video instead.")

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle
