import os
from pathlib import Path

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen2_5_VLProcessor

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.sa2va.models import Sa2VAModel, SAM2TrainRunner
from projects.sa2va.datasets import (
    Sa2VARefCOCODataset,
    Sa2VAEntitySegDataset,
    sa2va_collect_fn_multitask,
)
from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA
from projects.sa2va.models.mllm.qwenvl import Qwen2_5_VL
from projects.sa2va.glamm_train_hook import GLaMMTrainFinalHook

run_name = "Sa2VA_7B_RefCOCO_EntitySeg"

REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_DIR = os.path.expanduser(os.environ.get("WORK_DIR", str(REPO_ROOT)))
DATA_DIR = os.path.expanduser(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))

work_dir = f"{WORK_DIR}/sa2va_work_dirs/{run_name}"

path = 'Qwen/Qwen2.5-VL-7B-Instruct'
pretrained_pth = f"{WORK_DIR}/sa2va_qwen2.5_7b.pth"
sam2_ckpt = f"{WORK_DIR}/sam2-hiera-large/sam2_hiera_large.pt"

load_weights_only = f"{WORK_DIR}/maoam_ckpts/sa2va/mp_rank_00_model_states.pt"

prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

batch_size = 1
accumulative_counts = 4
dataloader_num_workers = 16
max_epochs = 5
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_epochs = 0.05

save_steps = 2000
save_total_limit = 2

wandb_project = "MatSeg_Siggraph2026"
wandb_offline = False
no_wandb = False
no_save = False
save_every = 1
val_freq = 0
viz_interval_mult = 1
auto_resume = False

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=Sa2VAModel,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    fix_number=1,
    loss_sample_points=True,
    frozen_sam2_decoder=False,
    arch_type='qwen',
    weight_star=1.0,
    weight_referring=1.0,
    weight_vqa=0.0,
    mllm=dict(
        type=Qwen2_5_VL,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=['lm_head', 'embed_tokens'],
            target_modules=None,
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
        ckpt_path=sam2_ckpt,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5)
)


sa2va_glamm_default_dataset_configs = dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    prompt_template=prompt_template,
    max_length=max_length,
    arch_type='qwen',
    preprocessor=dict(
        type=Qwen2_5_VLProcessor.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
    )
)

refcoco_train_dataset = dict(
    type=Sa2VARefCOCODataset,
    name='GLaMM_RefCOCO',
    dataset_dir=DATA_DIR,
    refer_segm_data='refcoco||refcoco+||refcocog',
    split='train',
    validation=False,
    random_sampling=True,
    overfit=False,
    num_classes_per_sample=1,
    image_size=1024,
    global_image_encoder='openai/clip-vit-large-patch14-336',
    repeats=1.0,
    **sa2va_glamm_default_dataset_configs,
)

entityseg_train_dataset = dict(
    type=Sa2VAEntitySegDataset,
    name='GLaMM_EntitySeg',
    samples_json=f"{DATA_DIR}/entityseg/entityseg_insseg_train.json",
    stage='train',
    overfit=False,
    global_image_encoder='openai/clip-vit-large-patch14-336',
    repeats=1.0,
    **sa2va_glamm_default_dataset_configs,
)

train_dataset = dict(
    type=ConcatDatasetSa2VA,
    datasets=[refcoco_train_dataset, entityseg_train_dataset],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=sa2va_collect_fn_multitask)
)

refcoco_val_dataset = dict(
    type=Sa2VARefCOCODataset,
    name='GLaMM_RefCOCO_val',
    dataset_dir=DATA_DIR,
    refer_segm_data='refcoco||refcoco+||refcocog',
    split='val',
    validation=True,
    random_sampling=False,
    overfit=False,
    num_classes_per_sample=1,
    image_size=1024,
    global_image_encoder='openai/clip-vit-large-patch14-336',
    repeats=1.0,
    **sa2va_glamm_default_dataset_configs,
)

entityseg_val_dataset = dict(
    type=Sa2VAEntitySegDataset,
    name='GLaMM_EntitySeg_val',
    samples_json=f"{DATA_DIR}/entityseg/entityseg_insseg_val.json",
    stage='val',
    overfit=False,
    global_image_encoder='openai/clip-vit-large-patch14-336',
    repeats=1.0,
    **sa2va_glamm_default_dataset_configs,
)

val_dataset = dict(
    type=ConcatDatasetSa2VA,
    datasets=[refcoco_val_dataset, entityseg_val_dataset],
)

glamm_val_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=val_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type=sa2va_collect_fn_multitask),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
val_dataloader = None
val_cfg = None
val_evaluator = None

custom_hooks = [
    dict(
        type=GLaMMTrainFinalHook,
        wandb_project=wandb_project,
        wandb_offline=wandb_offline,
        no_wandb=no_wandb,
        no_save=no_save,
        save_every=save_every,
        val_freq=val_freq,
        viz_interval_mult=viz_interval_mult,
        auto_resume=auto_resume,
        load_weights_only=load_weights_only,
        val_dataloader=glamm_val_dataloader,
    )
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=None,
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
