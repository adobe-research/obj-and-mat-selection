import torch


def load_mp_rank_checkpoint(path: str):
    """
    Load a DeepSpeed ZeRO checkpoint shard (e.g. mp_rank_00_model_states.pt)
    and return the raw state dict plus any metadata.
    """
    ckpt = torch.load(path, map_location="cpu")
    if "module" in ckpt:
        state_dict = ckpt["module"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    return ckpt, state_dict


if __name__ == "__main__":
    import sys as _sys
    ckpt_path = _sys.argv[1] if len(_sys.argv) > 1 else None
    if not ckpt_path:
        print("Usage: python check_load.py <mp_rank_00_model_states.pt>")
        _sys.exit(1)
    ckpt_all, model_state = load_mp_rank_checkpoint(ckpt_path)
    print(f"Loaded {len(model_state)} tensors")
