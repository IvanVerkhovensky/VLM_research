#!/usr/bin/env bash
set -e
python -m audit_smolvla.smoke
python -m audit_smolvla.audit_language --n_frames 200
python -m audit_smolvla.audit_ood --n_frames 200 --perturb brightness
python -m audit_smolvla.audit_ood --n_frames 200 --perturb gaussian_noise
python -m audit_smolvla.audit_ood --n_frames 200 --perturb occlusion
