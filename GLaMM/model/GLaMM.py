from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)
from model.SAM import build_sam_vit_h








def calculate_dice_loss(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    mask_count: float,
    scale_factor=1000,
    epsilon=1e-6,
):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)
    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (
        ground_truth / scale_factor
    ).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(
    predictions: torch.Tensor, targets: torch.Tensor, mask_count: float
):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class GLaMMBaseModel:
    def __init__(self, config, **kwargs):
        super(GLaMMBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(
            self.config, "out_dim", kwargs.get("out_dim", 512)
        )

        self.initialize_glamm_model(self.config)

    def initialize_glamm_model(self, config):
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self._configure_grounding_encoder(config)

        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()


class GLaMMModel(GLaMMBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(GLaMMModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class GLaMMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = GLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get(
            "vision_module", "openai/clip-vit-large-patch14-336"
        )
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        torch.cuda.empty_cache()
        with torch.no_grad():
            return self.model.grounding_encoder.image_encoder(pixel_values)

    def forward(self, **kwargs):
        return (
            super().forward(**kwargs)
            if "past_key_values" in kwargs
            else self.model_forward(**kwargs)
        )







    def model_forward(
        self,
        images_star: torch.FloatTensor = None,
        images_without_star: torch.FloatTensor = None,
        grounding_enc_images: torch.FloatTensor = None,
        masks_list: List[torch.FloatTensor] = None,
        orig_size: List[tuple] = None,
        resize_list: List[tuple] = None,
        inference: bool = False,
        **kwargs,
    ):
        """
        Unified forward pass for multi-task learning.
        Processes all tasks in a single forward call.
        """
        count = 0
        star_dict = kwargs.get("star", None)
        if star_dict is not None:
            count += 1
        referring_dict = kwargs.get("referring", None)
        if referring_dict is not None:
            count += 1
        vqa_dict = kwargs.get("vqa", None)
        if vqa_dict is not None:
            count += 1

        grounding_embeddings = (
            self.get_grounding_encoder_embs(grounding_enc_images)
            if grounding_enc_images is not None
            else None
        )

        tasks = []
        task_names = []
        task_uses_grounding = []

        if star_dict is not None or vqa_dict is not None:
            reference_image = images_star
        else:
            reference_image = images_without_star

        task_images = torch.zeros(
            (reference_image.shape[0] * count, *(reference_image.shape[1:])),
            dtype=reference_image.dtype,
            device=reference_image.device,
        )

        num_tasks = 0
        if star_dict is not None:
            tasks.append(star_dict)
            task_names.append("star")
            task_uses_grounding.append(True)
            task_images[
                num_tasks
                * reference_image.shape[0] : (num_tasks + 1)
                * reference_image.shape[0]
            ] = images_star
            num_tasks += 1

        if referring_dict is not None:
            tasks.append(referring_dict)
            task_names.append("referring")
            task_uses_grounding.append(True)
            task_images[
                num_tasks
                * reference_image.shape[0] : (num_tasks + 1)
                * reference_image.shape[0]
            ] = images_without_star
            num_tasks += 1

        if vqa_dict is not None:
            tasks.append(vqa_dict)
            task_names.append("vqa")
            task_uses_grounding.append(False)
            task_images[
                num_tasks
                * reference_image.shape[0] : (num_tasks + 1)
                * reference_image.shape[0]
            ] = images_star
            num_tasks += 1

        if num_tasks == 0:
            return {"star": None, "referring": None, "vqa": None}

        all_input_ids = [td["input_ids"] for td in tasks]
        all_labels = [td["labels"] for td in tasks]
        all_attn_masks = [td["attention_masks"] for td in tasks]

        task_images_list = []
        task_boundaries = [0]
        cursor = 0
        max_seq_len = max(ids.shape[1] for ids in all_input_ids)

        for i, td in enumerate(tasks):
            ids_i = all_input_ids[i]
            labels_i = all_labels[i]
            attn_i = all_attn_masks[i]
            n_seq = int(ids_i.shape[0])

            if task_names[i] in ("star", "vqa"):
                img_src = images_star
            else:
                img_src = images_without_star

            offset = td.get("offset", None)
            if offset is None:
                task_img = img_src
            else:
                off = offset.detach().cpu().tolist()
                reps = [int(off[j + 1] - off[j]) for j in range(len(off) - 1)]
                task_img = torch.cat(
                    [img_src[j : j + 1].repeat(max(0, r), 1, 1, 1) for j, r in enumerate(reps)],
                    dim=0,
                )

            if task_img.shape[0] != n_seq:
                raise RuntimeError(
                    f"Task {task_names[i]!r} has {n_seq} sequences but {task_img.shape[0]} images. "
                    f"Offsets likely mismatched."
                )
            task_images_list.append(task_img)

            task_boundaries.append(cursor + n_seq)
            cursor += n_seq

        total_seqs = int(cursor)
        task_images = torch.cat(task_images_list, dim=0)

        padded_input_ids = torch.full(
            (total_seqs, max_seq_len),
            self.config.pad_token_id,
            dtype=all_input_ids[0].dtype,
            device=all_input_ids[0].device,
        )
        padded_labels = torch.full(
            (total_seqs, max_seq_len),
            -100,
            dtype=all_labels[0].dtype,
            device=all_labels[0].device,
        )
        padded_attn_masks = torch.zeros(
            (total_seqs, max_seq_len),
            dtype=torch.bool,
            device=all_attn_masks[0].device,
        )

        for i in range(num_tasks):
            start = task_boundaries[i]
            end = task_boundaries[i + 1]
            seq_len_i = int(all_input_ids[i].shape[1])
            padded_input_ids[start:end, :seq_len_i] = all_input_ids[i]
            padded_labels[start:end, :seq_len_i] = all_labels[i]
            padded_attn_masks[start:end, :seq_len_i] = all_attn_masks[i]

        output = super().forward(
            images=task_images,
            attention_mask=padded_attn_masks,
            input_ids=padded_input_ids,
            labels=padded_labels,
            output_hidden_states=True,
            bboxes=None,
        )

        output_hidden_states = output.hidden_states
        if not isinstance(output_hidden_states, (tuple, list)):
            output_hidden_states = [output_hidden_states]
        output_hidden_states = torch.stack(output_hidden_states, dim=0)
        results = {}

        for i, task_name in enumerate(task_names):
            start_idx = task_boundaries[i]
            end_idx = task_boundaries[i + 1]

            task_input_ids = padded_input_ids[start_idx:end_idx]
            task_hidden_states = output_hidden_states[:, start_idx:end_idx]

            task_labels = padded_labels[start_idx:end_idx]
            task_output_logits = output.logits[start_idx:end_idx]  # Slice batch first!

            seq_len = task_labels.shape[1]
            task_output_logits = task_output_logits[:, -seq_len:, :]

            shift_logits = task_output_logits[:, :-1, :].contiguous()
            shift_labels = task_labels[:, 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)

            loss_fct = nn.CrossEntropyLoss()
            task_ce_loss = loss_fct(shift_logits, shift_labels)

            pred_masks = None
            mask_bce_loss = torch.tensor(0.0, device=task_ce_loss.device)
            mask_dice_loss = torch.tensor(0.0, device=task_ce_loss.device)

            if task_uses_grounding[i] and grounding_embeddings is not None:
                seg_token_mask = self._create_seg_token_mask(task_input_ids)

                task_dict = tasks[i]
                offset = task_dict.get("offset")

                hidden_states, pred_embeddings = self._process_hidden_states(
                    task_hidden_states, seg_token_mask, offset
                )

                pred_masks = self._generate_and_postprocess_masks(
                    pred_embeddings, grounding_embeddings, resize_list, orig_size
                )

                if pred_masks is not None and masks_list is not None:
                    gt_masks = torch.stack(masks_list, dim=0)

                    batch_size = pred_masks.shape[0]
                    instances_per_sample = pred_masks.shape[1]
                    num_masks = batch_size * instances_per_sample

                    flat_pred = pred_masks.view(num_masks, *pred_masks.shape[2:])
                    flat_gt = gt_masks.view(num_masks, *gt_masks.shape[2:])

                    mask_bce_loss = compute_sigmoid_cross_entropy(
                        flat_pred, flat_gt, mask_count=float(instances_per_sample)
                    )
                    mask_dice_loss = calculate_dice_loss(
                        flat_pred, flat_gt, mask_count=float(instances_per_sample)
                    )

                    mask_bce_loss = (
                        self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
                    )
                    mask_dice_loss = (
                        self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
                    )

            ce_loss = task_ce_loss * self.ce_loss_weight
            mask_loss = mask_bce_loss + mask_dice_loss
            total_loss = ce_loss + mask_loss

            results[task_name] = {
                "loss": total_loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
                "pred_masks": pred_masks,
                "gt_masks": masks_list if task_uses_grounding[i] else None,
            }

        for task_name in ["star", "referring", "vqa"]:
            if task_name not in results:
                results[task_name] = None

        return results

    def _create_seg_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.seg_token_idx
        return torch.cat(
            [
                torch.zeros((mask.shape[0], 575)).bool().cuda(),
                mask,
                torch.zeros((mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(
            length, -1, -1, -1
        ).contiguous()

        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i : i + 1],
                attention_mask=attention_masks[i : i + 1],
                input_ids=input_ids[i : i + 1],
                output_hidden_states=True,
            )
            output_hidden_states.append(output_i.hidden_states[-1])
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states



    def _process_hidden_states(
        self, output_hidden_states, seg_token_mask, offset, infer=False
    ):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        if seg_token_mask.dim() == 2 and last_hidden_state.dim() == 2:
            if (
                seg_token_mask.shape[0] == 1
                and last_hidden_state.shape[0] == seg_token_mask.shape[1]
            ):
                last_hidden_state = last_hidden_state.unsqueeze(0)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        segment_sizes = seg_token_offset[1:] - seg_token_offset[:-1]
        pred_embeddings_list = torch.split(
            pred_embeddings, segment_sizes.tolist(), dim=0
        )
        pred_embeddings_list = torch.stack(pred_embeddings_list, dim=0)
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(
        self, pred_embeddings, image_embeddings, resize_list, orig_size, infer=False
    ):
        if pred_embeddings is None:
            return None

        if pred_embeddings.dim() != 3:
            raise ValueError(
                f"Expected pred_embeddings to have shape [B,K,D], got {tuple(pred_embeddings.shape)}"
            )

        b, k, d = pred_embeddings.shape
        pred_flat = pred_embeddings.reshape(b * k, 1, d)

        if image_embeddings is None:
            return None
        img_flat = (
            image_embeddings.unsqueeze(1)
            .expand(b, k, *image_embeddings.shape[1:])
            .reshape(b * k, *image_embeddings.shape[1:])
        )

        sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=pred_flat,
        )

        low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
            image_embeddings=img_flat,
            image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        pred_masks = self.model.grounding_encoder.postprocess_masks(
            low_res_masks,
            input_size=resize_list[0],
            original_size=orig_size,
        )  # [B*K, 1, H, W]

        pred_masks = pred_masks.squeeze(1).reshape(b, k, *pred_masks.shape[-2:])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output):
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        if pred_masks is not None:
            gt_masks = torch.stack(masks_list, dim=0)

            batch_size = pred_masks.shape[0]
            instances_per_sample = pred_masks.shape[1]
            num_masks = batch_size * instances_per_sample

            flat_pred = pred_masks.view(num_masks, *pred_masks.shape[2:])
            flat_gt = gt_masks.view(num_masks, *gt_masks.shape[2:])

            mask_bce_loss = compute_sigmoid_cross_entropy(
                flat_pred, flat_gt, mask_count=float(instances_per_sample)
            )
            mask_dice_loss = calculate_dice_loss(
                flat_pred, flat_gt, mask_count=float(instances_per_sample)
            )

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        total_loss = ce_loss + mask_loss
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        global_enc_images,
        grounding_enc_images,
        input_ids,
        resize_list,
        orig_size,
        max_tokens_new=32,
        bboxes=None,
    ):
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images,
                input_ids=input_ids,
                bboxes=bboxes,
                max_new_tokens=max_tokens_new,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    torch.zeros(
                        (seg_token_mask.shape[0], 575), dtype=torch.bool
                    ).cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings,
                image_embeddings,
                resize_list,
                orig_size,
                infer=True,
            )
        return generated_output_ids, pred_masks
