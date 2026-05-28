<div align="center">

# MAOAM: Unified Object &amp; Material Selection with Vision-Language Models

<p>
  <a href="https://jadenpark0.github.io/"><b>Jaden Park</b></a><sup>1</sup>
  &nbsp;·&nbsp;
  <a href="https://valentin.deschaintre.fr/"><b>Valentin Deschaintre</b></a><sup>2</sup>
  &nbsp;·&nbsp;
  <a href="https://research.adobe.com/person/jason-kuen/"><b>Jason Kuen</b></a><sup>3</sup>
  &nbsp;·&nbsp;
  <a href="https://kangning-liu.github.io/"><b>Kangning Liu</b></a><sup>3</sup>
  &nbsp;·&nbsp;
  <a href="https://iliyan.com/"><b>Iliyan Georgiev</b></a><sup>2</sup>
  <br>
  <a href="https://krsingh.cs.ucdavis.edu/"><b>Krishna Kumar Singh</b></a><sup>3</sup>
  &nbsp;·&nbsp;
  <a href="https://pages.cs.wisc.edu/~yongjaelee/"><b>Yong Jae Lee</b></a><sup>3</sup>
  &nbsp;·&nbsp;
  <a href="https://mfischer-ucl.github.io/"><b>Michael Fischer</b></a><sup>2</sup>
</p>

<sup>1</sup>&nbsp;University of Wisconsin–Madison &nbsp;·&nbsp;
<sup>2</sup>&nbsp;Adobe Research, UK &nbsp;·&nbsp;
<sup>3</sup>&nbsp;Adobe Research, USA

<br>

<img src="docs/siggraph_logo.png" width="180" alt="SIGGRAPH">

**SIGGRAPH Conference Papers '26**

<p>
  <a href="https://arxiv.org/abs/2606.04880">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.04880-b31b1b.svg">
  </a>
  <a href="https://doi.org/10.1145/3799902.3811186">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-ACM%20DL-b31b1b.svg">
  </a>
  <a href="https://jadenpark0.github.io/project_pages/maoam/">
    <img alt="Project page" src="https://img.shields.io/badge/Project-Page-1f6feb.svg">
  </a>
  <a href="https://huggingface.co/jpark677/maoam_ckpts">
    <img alt="Weights" src="https://img.shields.io/badge/%F0%9F%A4%97-Weights-ffd21e.svg">
  </a>
  <a href="https://huggingface.co/datasets/jpark677/maoam_data">
    <img alt="Test data" src="https://img.shields.io/badge/%F0%9F%A4%97-Test%20data-ffd21e.svg">
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Adobe%20Research-brightgreen.svg">
  </a>
</p>

<img src="docs/teaser.png" width="92%" alt="MAOAM teaser">

</div>

<br>

> **TL;DR** — MAOAM is a unified model that selects **objects or materials** from
> **text prompts, clicks, or both**.

## Contents

- [✨ Capabilities](#-capabilities)
- [🧠 Architecture and data](#-architecture-and-data)
- [⚙️ Setup](#%EF%B8%8F-setup)
- [Demo](#demo)
- [📊 Evaluation](#-evaluation)
- [Data](#data)
- [🙏 Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## ✨ Capabilities

See the
[project page](https://jadenpark0.github.io/project_pages/maoam/#capabilities)
for the full set with larger figures.

---

## 🧠 Architecture and data

<div align="center">
<img src="docs/method.png" width="92%" alt="MAOAM architecture">
</div>

Given an input image, MAOAM takes a task prompt specifying the selection
criteria (objects or materials) alongside a user prompt in click or text. If
a click is provided, stars are overlaid onto the image as visual cues.

### Material mask sources

Training a unified selection model requires material datasets with dense mask
annotations and rich text descriptions. Existing material datasets either
lack textual descriptions, are flat material maps, or are tied to a specific
domain. We therefore collect new material data from both real and synthetic
sources to capture natural diversity alongside precise, controlled
annotations.

<div align="center">
<img src="docs/dataset.png" width="92%" alt="Example dataset annotations">
</div>

| Set | Masks | Images | Source |
|:--|:--:|:--:|:--|
| **RealMat** | ~49K | ~8K | Real images from [Pexels](https://www.pexels.com), hand-annotated with material masks. |
| **SynMat**  | ~55K | ~5.5K | Blender renders from 132 [Evermotion](https://www.evermotion.org) scenes with semantically valid material assignments (a sofa is leather, not stone). |
| **SAMa**    | ~3.3K | ~1.3K | Multi-view video frames from [SAMa](https://mfischer-ucl.github.io/sama/); assignments aren't always semantically meaningful but are consistent across object parts. |

### Description generation

We use **Qwen3-VL-235B-A22B-Thinking** with Set-of-Marks prompting to generate candidate annotations, then incorporate quality control through model-based verification.

For each marked region we generate three types of descriptions: **(i)** a
short material description paired with an entity label (e.g.&nbsp;*"the white
ceramic chair"*); **(ii)** a short material description with spatial
information, absolute (*"bottom right corner"*) or relative (*"above the
table"*); and **(iii)** a longer, self-contained material description that
does not rely on context. We sample **6 variants** per region from 10 to 50
words. Model-based verification fixes incorrect grounding and instruction-following.

### VQA generation with hard-negative mining

Training on VQA encourages fine-grained material understanding through
reasoning in text. We formulate two variants of a 4-way multiple-choice task:
**Q1** samples distractors from other regions in the same image (or other
images, if too few materials are present); **Q2** introduces hard-negative
mining where the answer's description is paraphrased into a visually
plausible but incorrect alternative — e.g.&nbsp;*"brown wood with dark
streaks"* &rarr; *"horizontally grained light wood"*.

<div align="center">
<img src="docs/vqa_gen.png" width="92%" alt="VQA generation pipeline">
</div>

### Training data composition

To train a unified model for both object- and material-level selection, we
additionally incorporate publicly available object segmentation datasets:
**RefCOCO / RefCOCO+ / RefCOCOg** for text-based object selection, and
**EntitySeg** for click-based object selection. The combined corpus contains
**~190K training samples** at an approximate **1:1 ratio** of material- to
object-centric data, spanning diverse selection prompts and criteria.

---

## ⚙️ Setup

We provide two MAOAM backbones with identical interfaces:

| Backbone | VLM | Mask head |
|:-:|:-:|:-:|
| **MAOAM-GLaMM** | LLaVA-Llama (GLaMM) | SAM ViT-H |
| **MAOAM-Sa2VA** | Qwen2.5-VL-7B       | SAM2 Hiera-L |

> **Hardware.** **MAOAM-Sa2VA requires an 80 GB A100** at the reference
> setting `qwen_max_pixels = 2048×28×28`. Lower `qwen_max_pixels` in the
> Sa2VA cfg to fit on a smaller card.

The two backends have independent dependency stacks (different Python /
PyTorch / `transformers` versions), so we keep them in **separate virtual
envs** managed by [`uv`](https://docs.astral.sh/uv/). Install `uv` once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the env for whichever backend you need:

```bash
# MAOAM-GLaMM   (Python 3.10, torch 2.4 + cu124)
cd GLaMM && uv sync

# MAOAM-Sa2VA   (Python 3.11, torch 2.6 + cu124, transformers 4.57)
cd Sa2VA && uv sync --extra latest
```

Each env lives under the backend folder as `.venv/` — activate with
`source .venv/bin/activate` or prefix any command with `uv run`.

### Pretrained weights

```bash
mkdir -p weights

# MAOAM finetuned weights from HuggingFace
hf download jpark677/maoam_ckpts --local-dir weights
#   weights/glamm/mp_rank_00_model_states.pt
#   weights/sa2va/mp_rank_00_model_states.pt

# Frozen vision backbones
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/
mkdir -p weights/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P weights/sam2/
```

Base LLMs are pulled from HuggingFace on first run:
`MBZUAI/GLaMM-GranD-Pretrained` and `Qwen/Qwen2.5-VL-7B-Instruct`.

### Code structure

```
obj-and-mat-selection/        ← repo root
├── README.md
├── utils/hm_utils.py
│
├── GLaMM/
│   ├── pyproject.toml        ← GLaMM venv (Python 3.10)
│   ├── demo.py
│   ├── check_load.py
│   ├── model/
│   ├── dataset/
│   └── tools/
│
├── Sa2VA/
│   ├── pyproject.toml        ← Sa2VA venv (Python 3.11)
│   ├── demo.py
│   ├── projects/sa2va/
│   │   ├── configs/
│   │   ├── models/sa2va.py
│   │   ├── datasets/
│   │   ├── gradio/app_utils.py
│   │   └── evaluation/sa2va_eval_all.py
│   ├── third_parts/sam2/
│   └── vlm/
│
├── docs/
└── weights/
    ├── glamm/mp_rank_00_model_states.pt
    ├── sa2va/mp_rank_00_model_states.pt
    ├── sam2/sam2_hiera_large.pt
    └── sam_vit_h_4b8939.pth
```

---

## Demo

A Gradio demo per backend. Both commands must be run from the **repo root**
(`utils/hm_utils.py` is imported from there).

```bash
# MAOAM-GLaMM  (uses GLaMM/.venv)
GLaMM/.venv/bin/python GLaMM/demo.py \
    --model_path MBZUAI/GLaMM-GranD-Pretrained \
    --resume weights/glamm/mp_rank_00_model_states.pt \
    --vision_pretrained weights/sam_vit_h_4b8939.pth \
    --port 7860

# MAOAM-Sa2VA  (uses Sa2VA/.venv)
PYTHONPATH=Sa2VA Sa2VA/.venv/bin/python Sa2VA/demo.py \
    --cfg Sa2VA/projects/sa2va/configs/glamm_qwen25_7b_material_only.py \
    --resume weights/sa2va/mp_rank_00_model_states.pt \
    --sam2_ckpt weights/sam2/sam2_hiera_large.pt \
    --port 7861
```

Open `http://localhost:7860` / `http://localhost:7861`. Upload an image, click
up to 5 points to drop stars (optional), type a text prompt, hit **Submit**.
A download zip with the binary mask + cyan overlays is one click away.

| Mode | Click | Text | Notes |
|:--|:--:|:--:|:--|
| Click only         | &check; | – | uses the canonical material prompt |
| Text only          | – | &check; | original image, no star |
| Click &nbsp;+&nbsp; text | &check; | &check; | combines both cues (paper Fig.&nbsp;7) |

---

## 📊 Evaluation

Paper Table&nbsp;2: RefCOCO / RefCOCO+ / RefCOCOg val + test splits, EntitySeg val.
Both eval scripts live in their respective backend folders and run from the **repo root**.

Set the model location and, if your datasets are not under the repository's
`data/` directory, override the data location:

```bash
export WEIGHTS=/path/to/weights   # contains glamm/, sa2va/, sam2/, sam_vit_h_4b8939.pth
export DATA_ROOT=/path/to/data    # contains entityseg/ and Refer_Segm/
```

`DATA_ROOT` defaults to `<repo>/data`. Material datasets default to
`<DATA_ROOT>/maoam_data` and can be overridden independently with
`MATERIAL_DATA_ROOT`.

### MAOAM-GLaMM

```bash
# 4 GPUs — all datasets
PYTHONPATH=. GLaMM/.venv/bin/torchrun --nproc_per_node=4 \
    GLaMM/tools/eval_all.py \
    --model_path MBZUAI/GLaMM-GranD-Pretrained \
    --resume "$WEIGHTS/glamm/mp_rank_00_model_states.pt" \
    --vision_pretrained "$WEIGHTS/sam_vit_h_4b8939.pth" \
    --data_root "$DATA_ROOT"

# EntitySeg only (if RefCOCO data not yet downloaded)
PYTHONPATH=. GLaMM/.venv/bin/torchrun --nproc_per_node=4 \
    GLaMM/tools/eval_all.py \
    --model_path MBZUAI/GLaMM-GranD-Pretrained \
    --resume "$WEIGHTS/glamm/mp_rank_00_model_states.pt" \
    --vision_pretrained "$WEIGHTS/sam_vit_h_4b8939.pth" \
    --data_root "$DATA_ROOT" \
    --datasets entityseg
```

### MAOAM-Sa2VA

```bash
# 4 GPUs — all datasets
DATA_ROOT="$DATA_ROOT" PYTHONPATH=Sa2VA Sa2VA/.venv/bin/torchrun --nproc_per_node=4 \
    Sa2VA/projects/sa2va/evaluation/sa2va_eval_all.py \
    --resume "$WEIGHTS/sa2va/mp_rank_00_model_states.pt" \
    --work_dir "$(dirname "$WEIGHTS")"

# EntitySeg only
DATA_ROOT="$DATA_ROOT" PYTHONPATH=Sa2VA Sa2VA/.venv/bin/torchrun --nproc_per_node=4 \
    Sa2VA/projects/sa2va/evaluation/sa2va_eval_all.py \
    --resume "$WEIGHTS/sa2va/mp_rank_00_model_states.pt" \
    --work_dir "$(dirname "$WEIGHTS")" \
    --datasets entityseg
```

`--work_dir` for Sa2VA must be the **parent of `weights/`** (contains `weights/sam2/sam2_hiera_large.pt`).
Both scripts write per-split `*_summary.json` alongside the checkpoint.

---

## Data

We release a **10 % subset** of the material annotations from the paper &mdash;
with per-region text descriptions and VQA questions &mdash; across three sets:

| Split | (image, mat) pairs | Unique images | VQA questions | Descriptions |
|:--|--:|--:|--:|--:|
| **SynMat** | 5,431 | 2,582 | 10,862 | 32,586 |
| **RealMat** | 4,663 | 2,685 | 9,326 | 27,978 |
| **SAMa** | 330 | 239 | 658 | 1,974 |
| **Total** | **10,424** | **5,506** | **20,846** | **62,538** |

VQA has 2 questions per (image, mat) pair; descriptions have 6 variants per pair.

For object-centric comparison we
use **EntitySeg** val and **RefCOCO / RefCOCO+ / RefCOCOg** val + test
splits — these are not part of our release; the download script below pulls
them from upstream sources.

### Download

```bash
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"

# ── MAOAM test set (the released subset) ───────────────────────────────────
hf download jpark677/maoam_data --repo-type dataset --local-dir "$DATA_ROOT/maoam_data"

# ── RefCOCO / RefCOCO+ / RefCOCOg ──────────────────────────────────────────
# 1. Annotations: convert from HuggingFace parquet to REFER API format (~200 MB dl, ~180 MB output)
pip install pyarrow huggingface_hub   # if not already present
mkdir -p "$DATA_ROOT/Refer_Segm"
python tools/convert_refcoco_parquet.py --out_dir "$DATA_ROOT/Refer_Segm"

# 2. COCO 2014 train images  ~13 GB
wget http://images.cocodataset.org/zips/train2014.zip -P "$DATA_ROOT/"
mkdir -p "$DATA_ROOT/Refer_Segm/images/mscoco/images"
unzip "$DATA_ROOT/train2014.zip" -d "$DATA_ROOT/Refer_Segm/images/mscoco/images/"
rm "$DATA_ROOT/train2014.zip"

# ── EntitySeg ────────────────────────────────────────────────────────────────
# 1. Images via HuggingFace  ~55 GB
hf download qqlu1992/Adobe_EntitySeg --repo-type dataset --local-dir "$DATA_ROOT/entityseg"

# 2. COCO-style annotations  ~181 MB
mkdir -p "$DATA_ROOT/entityseg/annotations"
wget https://github.com/adobe-research/EntitySeg-Dataset/releases/download/v1.0/entityseg_insseg_train_annotations.json \
    -P "$DATA_ROOT/entityseg/annotations/"
wget https://github.com/adobe-research/EntitySeg-Dataset/releases/download/v1.0/entityseg_insseg_val_annotations.json \
    -P "$DATA_ROOT/entityseg/annotations/"
```

> **Note on EntitySeg images.** After the HuggingFace download, the three image
> archives (`entity_01_11580`, `entity_02_11598`, `entity_03_10049`) may arrive
> as tar files. Extract each so the final layout matches the tree below.

For reference, the upstream sources:

| Source | What | Size |
|:--|:--|--:|
| HF [`jxu124/refcoco`](https://huggingface.co/datasets/jxu124/refcoco), [`jxu124/refcocoplus`](https://huggingface.co/datasets/jxu124/refcocoplus), [`jxu124/refcocog`](https://huggingface.co/datasets/jxu124/refcocog) | RefCOCO / RefCOCO+ / RefCOCOg (converted via `tools/convert_refcoco_parquet.py`) | ~200 MB |
| `images.cocodataset.org` | COCO 2014 train images (used by RefCOCO) | ~13 GB |
| HF [`qqlu1992/Adobe_EntitySeg`](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg) | EntitySeg image archives | ~55 GB |
| [`adobe-research/EntitySeg-Dataset`](https://github.com/adobe-research/EntitySeg-Dataset/releases/tag/v1.0) v1.0 | EntitySeg COCO-style annotations | 181 MB |

Final layout (matches the dataloaders out of the box):

```
$DATA_ROOT/
├── maoam_data/                              # MAOAM release subset
│   ├── synmat_release.json
│   ├── synmat_descriptions.json
│   ├── synmat_vqa.json
│   ├── realmat_release.json
│   ├── realmat_descriptions.json
│   ├── realmat_vqa.json
│   ├── sama_release.json
│   ├── sama_descriptions.json
│   ├── sama_vqa.json
│   ├── synmat/
│   │   ├── images/   # PNG renders
│   │   └── masks/    # binary masks *_mat<id>.png
│   ├── realmat/
│   │   ├── images/
│   │   └── masks/
│   └── sama/
│       ├── images/
│       └── masks/
├── Refer_Segm/
│   ├── refcoco/  refcoco+/  refcocog/       # annotations + refs(...).p
│   └── images/mscoco/images/train2014/      # 82,783 .jpg, ~13 GB
└── entityseg/
    ├── annotations/                          # entityseg_insseg_{train,val}_annotations.json
    └── images/
        ├── entity_01_11580/images_merge/      # 11,580 .jpg
        ├── entity_02_11598/images/            # 11,598 .jpg
        └── entity_03_10049/images_03_10049/   # 10,049 .jpg
```

### File schemas

#### `{source}_release.json` — sample list

A flat JSON array. One entry per evaluated (image, material) pair.

```jsonc
[
  // SynMat entry
  {
    "source":    "synmat",
    "filepath":  "/synmat/AI09_002_frame0780_selection_materialistic.exr",
    "mat_id":    2,          // integer material ID
    "aggregate": false       // true → merge with visually identical sibling IDs
  },
  // RealMat entry
  {
    "source":     "realmat",
    "filepath":   "/realmat/material_20241203/000625.jpg",
    "mat_id":     1,
    "annotation": "ff98808fd6b067ba0981261732197b14"  // mask annotation hash
  },
  // SAMa entry
  {
    "source":   "sama",
    "filepath": "/sama/video16_frame0.exr",
    "mat_id":   1
  }
]
```

Image and mask paths are resolved as:

```
images/ {stem}.png
masks/  {stem}_mat{mat_id}.png
```

where `stem` is the filepath with its source prefix and extension stripped, and
any intermediate path separators replaced by `__`
(e.g. `/realmat/material_20241203/000625.jpg` → `material_20241203__000625`).

#### `{source}_descriptions.json` — 6 text descriptions per (image, mat) pair

```jsonc
{
  // key: basename for synmat/sama; relative subpath for realmat
  "AI09_002_frame0780_selection_materialistic.exr": {
    "2": {                           // mat_id as string
      "descriptions": [
        "short material label",
        "label with entity context",
        "description with absolute spatial location",
        "description with relative spatial location",
        "longer self-contained description",
        "paraphrase of the longer description"
      ]
    }
  }
}
```

> **Key convention.** For SynMat and SAMa the top-level key is `os.path.basename(filepath)`.
> For RealMat it is `filepath` with the `/realmat/` prefix stripped
> (e.g. `"material_20241203/000625.jpg"`), since basenames are not unique across subdirectories.

#### `{source}_vqa.json` — 2 × 4-way multiple-choice questions per (image, mat) pair

```jsonc
{
  "AI09_002_frame0780_selection_materialistic.exr": {
    "2": [                           // list of 2 questions
      {
        "A": "option text A",
        "B": "option text B",
        "C": "option text C",
        "D": "option text D",
        "answer": "C"                // correct letter
      },
      { "A": "...", "B": "...", "C": "...", "D": "...", "answer": "A" }
    ]
  }
}
```

Same key convention as `{source}_descriptions.json` above.

#### `Refer_Segm/` — RefCOCO / RefCOCO+ / RefCOCOg

```
Refer_Segm/<dataset>/
├── instances.json     # COCO-style {images, annotations, categories}
└── refs(<splitBy>).p  # pickled list[dict]; splitBy ∈ {unc, umd, google}
```

Each `refs(...).p` entry:

```python
{
    "image_id":    int,
    "ann_id":      int,          # FK -> mask in instances.json
    "ref_id":      int,
    "split":       "train" | "val" | "testA" | "testB" | "test",
    "sentences":   [{"sent_id": int, "sent": str, "tokens": [str], "raw": str}, ...],
    "sent_ids":    [int, ...],
    "category_id": int,
    "file_name":   "COCO_train2014_000000000009.jpg",
}
```

#### `entityseg/annotations/` — EntitySeg

COCO instance-segmentation format with two extra fields:

```jsonc
{
  "images":      [{"id": int, "file_name": "*.jpg", "height": int, "width": int}, ...],
  "annotations": [{
    "id":              int,
    "image_id":        int,
    "category_id":     int,          // remapped EntitySeg category
    "category_id_ori": int,          // original COCO/LVIS id
    "segmentation":    {"size": [H, W], "counts": "<RLE>"},
    "bbox":            [x, y, w, h],
    "area":            int,
    "iscrowd":         0 | 1,
    "attribute":       int | str,    // material/texture tag
    "remove":          0 | 1         // 1 = exclude from eval
  }, ...],
  "categories": [{"id": int, "name": str, "supercategory": str}, ...]
}
```

---

## Acknowledgements

This work was supported in part by NSF IIS2404180 and Institute of Information
&amp; communications Technology Planning &amp; Evaluation (IITP) grant funded
by the Korea government (MSIT) (No.&nbsp;2022-0-00871, Development of AI
Autonomy and Knowledge Enhancement for AI Agent Collaboration). The authors
would like to thank Sudeep Katakol for his help in data generation and Zijun
Wei, Yash Savani, and Soochahn Lee for helpful discussions.

This implementation builds on
[GLaMM](https://github.com/mbzuai-oryx/groundingLMM),
[Sa2VA](https://github.com/magic-research/Sa2VA),
[SAM](https://github.com/facebookresearch/segment-anything), and
[SAM2](https://github.com/facebookresearch/segment-anything-2).

---

## Citation

```bibtex
@inproceedings{park2026maoam,
  title     = {MAOAM: Unified Object and Material Selection with Vision-Language Models},
  author    = {Park, Jaden and Deschaintre, Valentin and Kuen, Jason and
               Liu, Kangning and Georgiev, Iliyan and Singh, Krishna Kumar and
               Lee, Yong Jae and Fischer, Michael},
  booktitle = {ACM SIGGRAPH 2026 Conference Papers},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3799902.3811186},
}
```
