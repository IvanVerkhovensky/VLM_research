from audit_smolvla.utils import (
    pick_device,
    load_smolvla,
    load_lerobot_dataset,
    ensure_task_text,
)


def main():
    model_id = "HuggingFaceVLA/smolvla_libero"
    dataset_id = "eunyoung927/smol-libero-v30"

    device = pick_device()
    print("device:", device)

    ds = load_lerobot_dataset(dataset_id)
    print("dataset:", dataset_id, "len:", len(ds))

    frame = dict(ds[0])
    frame = ensure_task_text(frame, ds)
    print("keys:", sorted(frame.keys())[:20], "...")

    lp = load_smolvla(model_id, device=device)
    a = lp.policy.select_action(lp.preprocess(frame))
    a = lp.postprocess(a)
    print("action shape:", getattr(a, "shape", None))
    print("ok")


if __name__ == "__main__":
    main()
