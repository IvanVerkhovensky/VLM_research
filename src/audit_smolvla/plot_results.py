import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load(p):
    with open(p, "r") as f:
        return json.load(f)

def num(x):
    
    return float(x) if x is not None else np.nan

def main():
    os.makedirs("figures", exist_ok=True)

    lang = load("outputs/language_audit.json")
    ood_b = load("outputs/ood_brightness.json")
    ood_n = load("outputs/ood_noise.json")
    ood_o = load("outputs/ood_occlusion.json")

    # Language sensitivity
    labels = ["Empty instruction", "Random instruction"]
    vals = [num(lang.get("mean_dist_empty")), num(lang.get("mean_dist_shuffle"))]

    plt.figure(figsize=(6,4))
    plt.bar(labels, vals)
    plt.ylabel("Mean L2(action difference)")
    plt.title("SmolVLA: language sensitivity (offline audit)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("figures/language_sensitivity.png", dpi=200)
    plt.close()

    # OOD sensitivity
    labels = ["brightness", "gaussian_noise", "occlusion"]
    vals = [num(ood_b.get("mean_dist")), num(ood_n.get("mean_dist")), num(ood_o.get("mean_dist"))]

    plt.figure(figsize=(6,4))
    plt.bar(labels, vals)
    plt.ylabel("Mean L2(action difference)")
    plt.title("SmolVLA: OOD visual sensitivity (offline audit)")
    plt.tight_layout()
    plt.savefig("figures/ood_sensitivity.png", dpi=200)
    plt.close()

    print("Saved figures to figures/")

if __name__ == "__main__":
    main()