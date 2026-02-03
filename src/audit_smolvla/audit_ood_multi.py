import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from audit_smolvla.utils import (
    apply_visual_perturbation_uint8,
    ensure_task_text,
    list_image_keys,
    load_lerobot_dataset,
    load_smolvla,
    pick_device,
    predict_action,
    sample_frame_indices,
    to_uint8_rgb,
)

PERTS = [
    ("brightness", "outputs/ood_brightness.json"),
    ("gaussian_noise", "outputs/ood_noise.json"),
    ("occlusion", "outputs/ood_occlusion.json"),
]


def to_torch_chw_float01(img_u8_hwc: np.ndarray) -> torch.Tensor:
    # uint8 HWC -> torch float32 CHW in [0,1]
    t = torch.from_numpy(img_u8_hwc).permute(2, 0, 1).contiguous().float() / 255.0
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--dataset_id", default="eunyoung927/smol-libero-v30")
    ap.add_argument("--n_frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = pick_device()
    lp = load_smolvla(args.model_id, device=device)
    ds = load_lerobot_dataset(args.dataset_id)

    idxs = sample_frame_indices(ds, n=args.n_frames, seed=args.seed)

    dists = {p: [] for p, _ in PERTS}
    skipped_no_images = 0
    skipped_bad_image = 0

    for i in tqdm(idxs, desc="OOD multi"):
        frame: Dict[str, Any] = ensure_task_text(dict(ds[i]), ds)

        img_keys = list_image_keys(frame)
        if not img_keys:
            skipped_no_images += 1
            continue

        
        imgs_u8 = {}
        try:
            for k in img_keys:
                imgs_u8[k] = to_uint8_rgb(frame[k])  
        except Exception:
            skipped_bad_image += 1
            continue

        base_seed = int(args.seed * 1_000_003 + i)

        
        a0 = predict_action(lp, frame, seed=base_seed)

        for p, _ in PERTS:
            frame_p = dict(frame)
            for k in img_keys:
                img_p_u8 = apply_visual_perturbation_uint8(imgs_u8[k], p, rng)

                
                t = to_torch_chw_float01(img_p_u8)
                if device.type == "mps":
                    
                    pass
                frame_p[k] = t

            a1 = predict_action(lp, frame_p, seed=base_seed)
            dists[p].append(float(np.linalg.norm(a0 - a1)))

    for p, out_path in PERTS:
        result = {
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "perturb": p,
            "n_requested": args.n_frames,
            "n_used": len(dists[p]),
            "n_skipped_no_images": skipped_no_images,
            "n_skipped_bad_image": skipped_bad_image,
            "device": str(device),
            "metric": "L2(action_orig - action_perturbed)",
            "mean_dist": float(np.mean(dists[p])) if dists[p] else None,
            "median_dist": float(np.median(dists[p])) if dists[p] else None,
            "note": "Torch CHW float images; deterministic per-frame seed to isolate visual effect.",
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print("saved", out_path, "n_used =", result["n_used"])


if __name__ == "__main__":
    main()
