import os
import sys
from typing import Optional

_GLAMM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_GLAMM_ROOT, ".."))
for _p in (_GLAMM_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.material_datasets.BaseMaterialDataset import BaseMaterialDataset
from utils.paths import material_data_root


class SamaDataset(BaseMaterialDataset):
    """
    SAMa material dataset using the maoam sama_release.json.
    Images and masks loaded from {base_data_dir}/sama/{images,masks}/.
    """

    def __init__(
        self,
        samples_json: Optional[str] = None,
        base_data_dir: Optional[str] = None,
        description_json: Optional[str] = None,
        vqa_json: Optional[str] = None,
        **kwargs,
    ):
        base_data_dir = str(base_data_dir or material_data_root())
        super().__init__(
            samples_json=samples_json or os.path.join(base_data_dir, "sama_release.json"),
            base_data_dir=base_data_dir,
            description_json=description_json or os.path.join(base_data_dir, "sama_descriptions.json"),
            vqa_json=vqa_json or os.path.join(base_data_dir, "sama_vqa.json"),
            **kwargs,
        )

    # _load_image_and_label inherited from BaseMaterialDataset (PNG-based)
