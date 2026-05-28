import torch

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from tools.glamm_eval_utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX

    non_empty_rounds = [r for r in rounds if r]
    for i, rou in enumerate(non_empty_rounds):
        parts = rou.split(sep)
        assert len(parts) == 2, (len(parts), rou)
        parts[0] += sep

        if DEFAULT_IMAGE_TOKEN in conversation:
            rou_ids = tokenizer_image_token(rou, tokenizer)
            inst_ids = tokenizer_image_token(parts[0], tokenizer)
        else:
            rou_ids = tokenizer(rou).input_ids
            inst_ids = tokenizer(parts[0]).input_ids

        if i > 0 and len(rou_ids) > 0 and rou_ids[0] == tokenizer.bos_token_id:
            rou_ids = rou_ids[1:]
        if i > 0 and len(inst_ids) > 0 and inst_ids[0] == tokenizer.bos_token_id:
            inst_ids = inst_ids[1:]

        round_len = len(rou_ids)
        instruction_len = len(inst_ids) - (2 if i == 0 else 1)

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len


def custom_collate_fn_multi(
    batch, tokenizer=None, use_mm_start_end=True, inference=False, local_rank=-1
):
    batch_size = len(batch)
    image_paths = [None] * batch_size
    image_star_list = [None] * batch_size
    image_no_star_list = [None] * batch_size
    grounding_images = [None] * batch_size
    masks_list = [None] * batch_size
    coords_list = [None] * batch_size
    sampled_classes_list = [None] * batch_size
    resize_list = [None] * batch_size

    task_data = {
        "star": {"convs": [], "questions": [], "count": 0, "offsets": [0]},
        "referring": {
            "convs": [],
            "questions": [],
            "descs": [],
            "count": 0,
            "offsets": [0],
        },
        "vqa": {
            "convs": [],
            "questions": [],
            "answers": [],
            "count": 0,
            "offsets": [0],
        },
    }

    global_enc_processor = batch[0]["global_enc_processor"]
    orig_size = batch[0]["orig_size"]

    for i, item in enumerate(batch):
        image_paths[i] = item["filepath"]
        image_star_list[i] = item["image_star"]
        image_no_star_list[i] = item["image_without_star"]
        grounding_images[i] = item["grounding_image"]
        masks_list[i] = item["masks"]
        coords_list[i] = item["coords"]
        sampled_classes_list[i] = item["sampled_classes"]
        resize_list[i] = item["orig_size"]

        for task in ["star", "referring", "vqa"]:
            if task == "vqa" and isinstance(item[task]["conversation"], list) and len(item[task]["conversation"]) > 1:
                conversations_list = item[task]["conversation"]
                questions_list = item[task]["question"]
                answers_list = item[task]["answer"]

                for conv, question, answer in zip(
                    conversations_list, questions_list, answers_list
                ):
                    if conv is not None:
                        task_data[task]["convs"].extend(conv)
                        task_data[task]["questions"].append(question)
                        task_data[task]["answers"].append(answer)
                        task_data[task]["count"] += len(conv)
                task_data[task]["offsets"].append(task_data[task]["count"])
            elif item[task]["conversation"] is not None:
                task_data[task]["convs"].extend(item[task]["conversation"])
                task_data[task]["questions"].append(item[task]["question"])
                task_data[task]["count"] += len(item[task]["conversation"])
                task_data[task]["offsets"].append(task_data[task]["count"])

                if task == "referring":
                    task_data[task]["descs"].append(item[task]["desc"])
                elif task == "vqa":
                    task_data[task]["answers"].append(item[task]["answer"])
            else:
                task_data[task]["offsets"].append(task_data[task]["count"])

    star_convs = task_data["star"]["convs"]
    star_questions = task_data["star"]["questions"]
    referring_convs = task_data["referring"]["convs"]
    referring_questions = task_data["referring"]["questions"]
    referring_descs = task_data["referring"]["descs"]
    vqa_convs = task_data["vqa"]["convs"]
    vqa_questions = task_data["vqa"]["questions"]
    vqa_answers = task_data["vqa"]["answers"]

    def _batch_process_images(image_list, processor, name):
        """Batch process images with single processor call"""
        if image_list[0] is None:
            return None
        image_list = [
            (img.clamp(0.0, 1.0) if isinstance(img, torch.Tensor) and img.is_floating_point() else img)
            for img in image_list
        ]
        return processor.preprocess(image_list, return_tensors="pt", do_rescale=False)[
            "pixel_values"
        ]

    clip_with = _batch_process_images(image_star_list, global_enc_processor, "star")
    clip_without = _batch_process_images(
        image_no_star_list, global_enc_processor, "no_star"
    )

    if grounding_images[0] is None:
        grounding_batch = None
    else:
        grounding_batch = torch.stack(grounding_images, dim=0)

    def _batch_replace_mm(convs_list, use_mm_start_end):
        """Batch replace multimodal tokens for all conversation lists"""
        if not use_mm_start_end or convs_list is None:
            return convs_list
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        return [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in convs_list]

    star_convs = _batch_replace_mm(star_convs, use_mm_start_end)
    referring_convs = _batch_replace_mm(referring_convs, use_mm_start_end)
    vqa_convs = _batch_replace_mm(vqa_convs, use_mm_start_end)

    def _tokenize_convs(convs):
        if not convs or len(convs) == 0:
            return None, None, None
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [tokenizer_image_token(c, tokenizer, return_tensors="pt") for c in convs],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attn = input_ids.ne(tokenizer.pad_token_id)
        labels = input_ids.clone()

        conv = conversation_lib.default_conversation.copy()
        sep = conv.sep + conv.roles[1] + ": "
        sep2 = conv.sep2
        for c, t in zip(convs, labels):
            _process_conversation(c, t, tokenizer, sep, sep2)

        truncate_len = tokenizer.model_max_length - 575
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            labels = labels[:, :truncate_len]
            attn = attn[:, :truncate_len]

        return input_ids, labels, attn

    star_inputs = _tokenize_convs(star_convs)
    ref_inputs = _tokenize_convs(referring_convs)
    vqa_inputs = _tokenize_convs(vqa_convs)

    return {
        "image_paths": image_paths,
        "images_star": clip_with,
        "images_without_star": clip_without,
        "grounding_enc_images": grounding_batch,
        "coords": coords_list,
        "sampled_classes": sampled_classes_list,
        "masks_list": masks_list,
        "orig_size": orig_size,
        "resize_list": None if resize_list[0] is None else resize_list,
        "inference": inference,
        "star": (
            None
            if len(star_questions) == 0
            else {
                "questions": star_questions,
                "convs": star_convs,
                "input_ids": None if star_inputs is None else star_inputs[0],
                "labels": None if star_inputs is None else star_inputs[1],
                "attention_masks": None if star_inputs is None else star_inputs[2],
                "offset": torch.LongTensor(task_data["star"]["offsets"]),
            }
        ),
        "referring": (
            None
            if len(referring_questions) == 0
            else {
                "questions": referring_questions,
                "convs": referring_convs,
                "desc": referring_descs,
                "input_ids": None if ref_inputs is None else ref_inputs[0],
                "labels": None if ref_inputs is None else ref_inputs[1],
                "attention_masks": None if ref_inputs is None else ref_inputs[2],
                "offset": torch.LongTensor(task_data["referring"]["offsets"]),
            }
        ),
        "vqa": (
            None
            if len(vqa_questions) == 0
            else {
                "questions": vqa_questions,
                "answers": vqa_answers,
                "convs": vqa_convs,
                "input_ids": None if vqa_inputs is None else vqa_inputs[0],
                "labels": None if vqa_inputs is None else vqa_inputs[1],
                "attention_masks": None if vqa_inputs is None else vqa_inputs[2],
                "offset": torch.LongTensor(task_data["vqa"]["offsets"]),
            }
        ),
    }
