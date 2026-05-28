from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
import time
from contextlib import contextmanager

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_BBOX_TOKEN = "<bbox>"


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name:>12} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("  ".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    def _to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda(non_blocking=True)
        if isinstance(obj, dict):
            return {kk: _to_cuda(vv) for kk, vv in obj.items()}
        if isinstance(obj, list):
            return [_to_cuda(ele) for ele in obj]
        if isinstance(obj, tuple):
            return tuple(_to_cuda(ele) for ele in obj)
        return obj

    for k, v in input_dict.items():
        input_dict[k] = _to_cuda(v)
    return input_dict


class TimingProfiler:
    """Minimal timing profiler for training phases."""

    def __init__(self, local_rank=0):
        self.local_rank = local_rank
        self.timings = {
            "overhead": [],
            "forward": [],
            "backward": [],
            "optimizer_step": [],
        }
        self._iter_start = None
        self._curr = {"forward": 0.0, "backward": 0.0, "optimizer_step": 0.0}

    def start_iteration(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._iter_start = time.time()
        self._curr = {"forward": 0.0, "backward": 0.0, "optimizer_step": 0.0}

    def end_iteration(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total = time.time() - self._iter_start if self._iter_start is not None else 0.0
        known = (
            self._curr["forward"]
            + self._curr["backward"]
            + self._curr["optimizer_step"]
        )
        overhead = max(0.0, total - known)
        self.timings["overhead"].append(overhead)
        self._iter_start = None

    @contextmanager
    def time_operation(self, operation_name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.time() - start
        self.timings[operation_name].append(duration)
        if operation_name in self._curr:
            self._curr[operation_name] += duration

    def print_timings(self, step):
        if self.local_rank == 0 and step % 3 == 0:  # Print every 3 steps
            avg = {k: np.mean(v) if v else 0 for k, v in self.timings.items()}
            print(
                f"Step {step} - Overhead: {avg['overhead']:.3f}s, "
                f"Forward: {avg['forward']:.3f}s, Backward: {avg['backward']:.3f}s, "
                f"Optimizer: {avg['optimizer_step']:.3f}s"
            )
