import argparse
import json
import os

import numpy as np

from audit_smolvla.utils import (
    ensure_task_text,
    load_lerobot_dataset,
    load_smolvla,
    pick_device,
    predict_action,
    sample_frame_indices,
)

FALLBACK_POOL = [
    "open the drawer",
    "close the drawer",
    "pick up the object",
    "place the object in the bowl",
    "move the object to the left",
    "move the object to the right",
    "push the object forward",
    "pull the object backward",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--dataset_id", default="eunyoung927/smol-libero-v30")
    ap.add_argument("--n_frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="outputs/language_audit.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = pick_device()
    lp = load_smolvla(args.model_id, device=device)
    ds = load_lerobot_dataset(args.dataset_id)

    idxs = sample_frame_indices(ds, n=args.n_frames, seed=args.seed)

    # pool of instructions
    pool = []
    for i in idxs[: min(1000, len(idxs))]:
        f = ensure_task_text(dict(ds[i]), ds)
        t = f.get("task", None)
        if isinstance(t, str) and t:
            pool.append(t)
    pool = sorted(set(pool))
    if len(pool) < 5:
        pool = FALLBACK_POOL

    d_repeat = []
    d_empty = []
    d_shuffle = []

    for i in idxs:
        frame = ensure_task_text(dict(ds[i]), ds)
        t = frame.get("task", None)
        if not (isinstance(t, str) and t):
            continue

        base_seed = int(args.seed * 1_000_003 + i)

        # repeatability (same seed -> should be ~0 if deterministic)
        a0 = predict_action(lp, frame, seed=base_seed)
        a0b = predict_action(lp, frame, seed=base_seed)
        d_repeat.append(float(np.linalg.norm(a0 - a0b)))

        # empty instruction (same base_seed to isolate language effect)
        f_empty = dict(frame)
        f_empty["task"] = ""
        a_empty = predict_action(lp, f_empty, seed=base_seed)
        d_empty.append(float(np.linalg.norm(a0 - a_empty)))

        # random instruction (same base_seed)
        f_rand = dict(frame)
        f_rand["task"] = rng.choice(pool)
        a_rand = predict_action(lp, f_rand, seed=base_seed)
        d_shuffle.append(float(np.linalg.norm(a0 - a_rand)))

    result = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "n_requested": args.n_frames,
        "n_used": len(d_repeat),
        "device": str(device),
        "metric": "L2(action_orig - action_variant)",
        "mean_dist_repeat": float(np.mean(d_repeat)) if d_repeat else None,
        "mean_dist_empty": float(np.mean(d_empty)) if d_empty else None,
        "mean_dist_shuffle": float(np.mean(d_shuffle)) if d_shuffle else None,
        "pool_size": len(pool),
        "note": "We fix a per-frame seed for orig/empty/shuffle to avoid sampling noise and isolate input effect.",
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
