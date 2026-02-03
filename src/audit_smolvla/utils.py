import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@dataclass
class LoadedPolicy:
    policy: SmolVLAPolicy
    preprocess: Any
    postprocess: Any
    device: torch.device
    model_id: str


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_smolvla(model_id: str, device: torch.device) -> LoadedPolicy:
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return LoadedPolicy(
        policy=policy, preprocess=preprocess, postprocess=postprocess, device=device, model_id=model_id
    )


def load_lerobot_dataset(repo_id: str) -> LeRobotDataset:
    return LeRobotDataset(repo_id)


def _to_int(x) -> Optional[int]:
    try:
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        if isinstance(x, np.ndarray):
            x = x.item()
        if torch.is_tensor(x):
            x = x.detach().cpu().item()
        return int(x)
    except Exception:
        return None


def task_index_to_text(ds: LeRobotDataset, task_index: int) -> Optional[str]:
    if not hasattr(ds, "meta") or not hasattr(ds.meta, "tasks"):
        return None
    tasks = ds.meta.tasks

    if hasattr(tasks, "columns") and "task_index" in tasks.columns and "task" in tasks.columns:
        row = tasks[tasks["task_index"] == task_index]
        if len(row) == 0:
            return None
        t = row.iloc[0]["task"]
        return str(t) if t is not None else None

    if isinstance(tasks, list):
        for r in tasks:
            if r.get("task_index") == task_index:
                return r.get("task")

    return None


def ensure_task_text(frame: Dict[str, Any], ds: Optional[LeRobotDataset]) -> Dict[str, Any]:
    out = dict(frame)
    t = out.get("task", None)
    if isinstance(t, str) and len(t) > 0:
        return out

    if ds is None:
        return out

    ti = _to_int(out.get("task_index", None))
    if ti is None:
        return out

    txt = task_index_to_text(ds, ti)
    if isinstance(txt, str) and len(txt) > 0:
        out["task"] = txt
    return out


def list_image_keys(frame: Dict[str, Any]) -> List[str]:
    return [k for k in frame.keys() if k.startswith("observation.images.")]


def to_uint8_rgb(img: Any) -> np.ndarray:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    try:
        from PIL import Image  # type: ignore
        if isinstance(img, Image.Image):
            img = np.array(img)
    except Exception:
        pass

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img)}")

    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {img.shape}")

    
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {img.shape}")

    if img.dtype == np.uint8:
        return img

    x = img.astype(np.float32)
    if np.nanmax(x) <= 1.0 + 1e-6:
        x = x * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def apply_visual_perturbation_uint8(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)

    if kind == "brightness":
        factor = rng.uniform(0.6, 1.4)
        x = x * factor
    elif kind == "gaussian_noise":
        sigma = rng.uniform(5.0, 25.0)
        x = x + rng.normal(0.0, sigma, size=x.shape)
    elif kind == "occlusion":
        h, w, _ = x.shape
        occ_h = int(rng.uniform(0.15, 0.35) * h)
        occ_w = int(rng.uniform(0.15, 0.35) * w)
        y0 = rng.integers(0, max(1, h - occ_h))
        x0 = rng.integers(0, max(1, w - occ_w))
        x[y0:y0 + occ_h, x0:x0 + occ_w, :] = rng.uniform(0, 255)
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")

    return np.clip(x, 0, 255).astype(np.uint8)


def reset_policy_state(policy: Any) -> None:
    
    if hasattr(policy, "reset") and callable(getattr(policy, "reset")):
        try:
            policy.reset()
            return
        except Exception:
            pass

    for m in ("reset_state", "reset_queues", "clear_queues"):
        if hasattr(policy, m) and callable(getattr(policy, m)):
            try:
                getattr(policy, m)()
                return
            except Exception:
                pass

    for attr in ("_action_queue", "_queues"):
        q = getattr(policy, attr, None)
        try:
            if hasattr(q, "clear"):
                q.clear()
        except Exception:
            pass


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict_action(lp: LoadedPolicy, frame: Dict[str, Any], seed: int) -> np.ndarray:
    
    set_seeds(seed)
    reset_policy_state(lp.policy)

    batch = lp.preprocess(dict(frame))
    action = lp.policy.select_action(batch)
    action = lp.postprocess(action)

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.array(action).reshape(-1)


def sample_frame_indices(dataset: LeRobotDataset, n: int, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(dataset), size=n).tolist()
