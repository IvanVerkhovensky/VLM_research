import argparse
import json
import os
from typing import Any, Dict

import numpy as np

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--dataset_id", default="eunyoung927/smol-libero-v30")
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", choices=["brightness", "gaussian_noise", "occlusion"], default="brightness")
    ap.add_argument("--out_json", default="outputs/ood_audit.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = pick_device()
    lp = load_smolvla(args.model_id, device=device)
    ds = load_lerobot_dataset(args.dataset_id)

    idxs = sample_frame_indices(ds, n=args.n_frames, seed=args.seed)

    dists = []
    n_skipped = 0

    for i in idxs:
        frame: Dict[str, Any] = ensure_task_text(dict(ds[i]), ds)
        img_keys = list_image_keys(frame)
        if not img_keys:
            n_skipped += 1
            continue

        a0 = predict_action(lp, frame)

        frame_p = dict(frame)
        ok = True
        for k in img_keys:
            try:
                img_u8 = to_uint8_rgb(frame_p[k])
                frame_p[k] = apply_visual_perturbation_uint8(img_u8, args.perturb, rng)
            except Exception:
                ok = False
                break

        if not ok:
            n_skipped += 1
            continue

        a1 = predict_action(lp, frame_p)
        dists.append(float(np.linalg.norm(a0 - a1)))

    result = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "perturb": args.perturb,
        "n_requested": args.n_frames,
        "n_used": len(dists),
        "n_skipped": n_skipped,
        "device": str(device),
        "metric": "L2(action_orig - action_perturbed)",
        "mean_dist": float(np.mean(dists)) if dists else None,
        "median_dist": float(np.median(dists)) if dists else None,
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
