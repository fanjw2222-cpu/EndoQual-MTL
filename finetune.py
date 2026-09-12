#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervised multitask fine-tuning.
Reconstructed implementation based on retained training materials and the
documented model design; this is not the recovered historical training script."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CONFIG = {
    "data_root": "data/labeled",
    "annotations_csv": "data/annotations/train_validation.csv",
    "ssl_checkpoint": "weights/ssl_encoder.pth",
    "output_root": "runs/finetune",
    "columns": {"filename": "filename", "patient_id": "patient_id", "lesion_id": "lesion_id",
                "split": "split", "main": "main_class", "structure_clarity": "structure_clarity",
                "stain_wash": "stain_wash", "focus": "blurriness", "brightness": "brightness"},
    "train_value": "train", "val_value": "val", "test_value": "test",
    
    
    "heldout_csvs": [],
    "check_image_files": True,
    "batch_size": 64, "num_workers": 0, "device": "auto", "seed": 42,
    "epochs": 30, "warmup_epochs": 3, "freeze_encoder_epochs": 3,
    "early_stopping_patience": 15,
    "encoder_lr": 2e-5, "heads_lr": 2e-4, "center_lr": 1e-4,
    "weight_decay": 0.1, "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
    "drop_path_rate": 0.3, "grad_clip_norm": 5.0,
    "focal_gamma": 2.0, "consistency_weight": 0.25, "center_weight": 0.02,
}
MODEL_NAME = "swin_large_patch4_window7_224.ms_in22k"
TASKS = ["main", "structure_clarity", "stain_wash", "focus", "brightness"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
STATUS = "RECONSTRUCTED_IMPLEMENTATION_NOT_HISTORICAL_SOURCE"
SOURCES = {
    "baseline_code_sha256": "4496c980455f1397ccfab611254335e9fb863d8050a096f6a6238ad74e5c42cc",
    "stage1_code_sha256": "389963bd1abac2511b7975126343e52a3d8fbb4aaa1914970fd1f3d5bf4d0cd2",
    "evaluation_code_sha256": "a871cde5f13540e61982027e1c47b0526d56c2ee8bc51a12bfefed6443208fc2",
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "numel") and value.numel() == 1:
        return json_safe(value.item())
    return str(value)


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with open(path, newline="", encoding=encoding) as stream:
                reader = csv.DictReader(stream)
                names = reader.fieldnames or []
                if not names or len(names) != len(set(names)):
                    raise ValueError(f"CSV header is empty or contains duplicate fields: {path}")
                rows = list(reader)
                if any(None in r or any(v is None for v in r.values()) for r in rows):
                    raise ValueError(f"CSV contains rows with inconsistent column counts: {path}")
                return names, rows
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to identify CSV encoding: {path}")


def label_code(value, main=False):
    value = str(value).strip()
    words = {"\u5dee": 0, "\u4e2d": 1, "\u597d": 2, "poor": 0, "fair": 1, "medium": 1, "good": 2,
             "poor quality": 0, "fair quality": 1, "medium quality": 1, "good quality": 2}
    if main and value.lower() in words:
        return words[value.lower()]
    try:
        number = float(value)
    except ValueError as ex:
        raise ValueError(f"Unrecognized score {value!r}") from ex
    if not math.isfinite(number) or number not in (0., 1., 2.):
        raise ValueError(f"Scores must be integer grades 0/1/2: {value!r}")
    return int(number)


def check_config(cfg):
    for key in ("batch_size", "epochs", "early_stopping_patience"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("warmup_epochs", "freeze_encoder_epochs", "num_workers"):
        if not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg["warmup_epochs"] >= cfg["epochs"] or cfg["freeze_encoder_epochs"] >= cfg["epochs"]:
        raise ValueError("Warm-up and frozen epochs must each be less than total epochs")
    for key in ("encoder_lr", "heads_lr", "center_lr", "adam_eps", "grad_clip_norm"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "focal_gamma", "consistency_weight", "center_weight"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 <= cfg["drop_path_rate"] < 1:
        raise ValueError("drop_path_rate must be in [0,1)")
    if len(set([cfg["train_value"], cfg["val_value"], cfg["test_value"]])) != 3:
        raise ValueError("Train, validation and test split values must differ")


def check_inputs(cfg):
    """Read existing annotations, never create a split or silently drop an image."""
    check_config(cfg)
    root = Path(cfg["data_root"]).expanduser().resolve()
    path = Path(cfg["annotations_csv"]).expanduser().resolve()
    ssl = Path(cfg["ssl_checkpoint"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found; update CONFIG['data_root']: {root}")
    if not path.is_file():
        raise FileNotFoundError(f"Annotation/split CSV not found; update CONFIG['annotations_csv']: {path}")
    if not ssl.is_file():
        raise FileNotFoundError(f"Stage-1 SSL weights not found: {ssl}; no initialization fallback is used")
    names, rows = read_rows(path)
    col = cfg["columns"]
    required = [col[k] for k in ["filename", "patient_id", "split"] + TASKS]
    missing = sorted(set(required) - set(names))
    if missing:
        raise ValueError(f"Missing annotation/split columns: {missing}; obtain patient_id from the original linkage records")
    groups = {cfg["train_value"]: [], cfg["val_value"]: [], cfg["test_value"]: []}
    seen = set()
    for line, row in enumerate(rows, 2):
        name = row[col["filename"]].strip().replace("\\", "/")
        patient = row[col["patient_id"]].strip()
        split = row[col["split"]].strip()
        if not name or name in seen:
            raise ValueError(f"Row {line} has an empty or duplicate filename: {name!r}")
        seen.add(name)
        if not patient or patient.lower() in {"nan", "none", "null", "na", "n/a"}:
            raise ValueError(f"Row {line} is missing a pseudonymous patient identifier")
        if split not in groups:
            raise ValueError(f"Row {line} has unknown split={split!r}; configure the actual train/val/test values")
        record = {"filename": name, "patient_id": patient,
                  "lesion_id": row.get(col["lesion_id"], "").strip(), "split": split}
        if split != cfg["test_value"]:
            try:
                record["labels"] = [label_code(row[col[t]], main=t == "main") for t in TASKS]
            except ValueError as ex:
                raise ValueError(f"CSV row {line}: {ex}") from ex
            fullpath = (root / name).resolve()
            if cfg["check_image_files"] and not fullpath.is_file():
                raise FileNotFoundError(f"Row {line}: image not found: {fullpath}")
            record["path"] = str(fullpath)
        groups[split].append(record)
    train = groups[cfg["train_value"]]; val = groups[cfg["val_value"]]
    if not train or not val:
        raise ValueError("The training or validation set is empty; provide the original split assignments")
    patient_sets = {k: {r["patient_id"] for r in v} for k, v in groups.items()}
    for a, b in [(cfg["train_value"], cfg["val_value"]),
                 (cfg["train_value"], cfg["test_value"]), (cfg["val_value"], cfg["test_value"])]:
        overlap = patient_sets[a] & patient_sets[b]
        if overlap:
            raise ValueError(f"{len(overlap)} patients overlap between {a}/{b}; review the original split assignments")
    info = {}
    for key, records in [("Training", train), ("Validation", val)]:
        counts = {t: [sum(r["labels"][i] == k for r in records) for k in range(3)]
                  for i, t in enumerate(TASKS)}
        if 0 in counts["main"]:
            raise ValueError(f"{key} does not cover all three main classes: {counts['main']}; the three-class selection rule cannot be applied")
        info[key] = {"images": len(records), "patients": len({r['patient_id'] for r in records}),
                     "lesions": len({(r['patient_id'], r['lesion_id']) for r in records})
                     if all(r['lesion_id'] for r in records) else None,
                     "class_counts_0_1_2": counts}
    heldout = []
    development_patients = patient_sets[cfg["train_value"]] | patient_sets[cfg["val_value"]]
    for extra in cfg["heldout_csvs"]:
        extra_path = Path(extra).expanduser().resolve()
        extra_names, extra_rows = read_rows(extra_path)
        if col["patient_id"] not in extra_names:
            raise ValueError(f"Held-out CSV missing {col['patient_id']}: {extra_path}")
        ids = {r[col["patient_id"]].strip() for r in extra_rows}
        if any(not x or x.lower() in {"nan", "none", "null", "na", "n/a"} for x in ids):
            raise ValueError(f"Held-out CSV has incomplete patient identifiers: {extra_path}")
        overlap = ids & development_patients
        if overlap:
            raise ValueError(f"Training/validation and held-out CSV share {len(overlap)} patient identifiers: {extra_path}")
        heldout.append({"path": str(extra_path), "sha256": sha256(extra_path), "patients": len(ids)})
    report = {"implementation_status": STATUS, "annotations_csv": str(path), "annotations_sha256": sha256(path),
              "data_root": str(root), "ssl_checkpoint": str(ssl), "ssl_size_bytes": ssl.stat().st_size,
              "train_validation_patient_overlap": 0, "groups": info,
              "test_rows_in_same_csv": len(groups[cfg['test_value']]), "heldout_checks": heldout,
              "heldout_patient_separation_checked": bool(groups[cfg['test_value']] or heldout),
              "limitations": ["Path checks do not establish image-content identity or historical split provenance.",
                              "No independent source confirms the original stage-2 training settings."]}
    return train, val, report


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Unrecognized checkpoint; expected a state_dict or dictionary containing model_state_dict")
    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    if not state or not all(isinstance(k, str) for k in state):
        raise ValueError("state_dict is empty or contains non-string keys")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state


def extract_ssl_encoder_state(checkpoint):
    state = extract_state_dict(checkpoint)
    if any(k.startswith(("main_head.", "clarity_head.", "task_weighter.")) for k in state):
        raise ValueError("This appears to be a final stage-2 model; stage-1 SSL encoder weights are required here")
    if any(k.startswith("encoder.") for k in state):
        unused = [k for k in state if not k.startswith("encoder.")]
        allowed = ("deg_type_head.", "severity_head.", "quality_head.")
        if any(not k.startswith(allowed) for k in unused):
            raise ValueError(f"Stage-1 checkpoint has unrecognized non-encoder keys: {unused[:8]}")
        return {k[8:]: v for k, v in state.items() if k.startswith("encoder.")}, "full_stage1_checkpoint", unused
    return state, "encoder_only_state_dict", []


def load_checkpoint(path):
    import torch
    if Path(path).suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    
    return torch.load(str(path), map_location="cpu", weights_only=True)


def build_model(cfg):
    import torch
    import torch.nn as nn
    import timm

    class DynamicTaskWeighting(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_vars = nn.Parameter(torch.zeros(5))

        def forward(self, losses):
            return sum(torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]
                       for i, loss in enumerate(losses))

    class MultiTaskSwinModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0,
                                             drop_path_rate=cfg["drop_path_rate"])
            d = self.encoder.num_features
            self.main_head = nn.Linear(d, 2)
            self.clarity_head = nn.Linear(d, 3)
            self.wash_head = nn.Linear(d, 3)
            self.blur_head = nn.Linear(d, 3)
            self.brightness_head = nn.Linear(d, 3)
            self.task_weighter = DynamicTaskWeighting()
            self.main_centers = nn.Parameter(torch.randn(3, d))
            nn.init.xavier_uniform_(self.main_centers)

        def forward(self, x):
            features = self.encoder(x)
            return (self.main_head(features), self.clarity_head(features), self.wash_head(features),
                    self.blur_head(features), self.brightness_head(features), features)

    return MultiTaskSwinModel().float()


def load_ssl(model, path):
    import torch
    raw = load_checkpoint(path)
    state, kind, unused = extract_ssl_encoder_state(raw)
    expected = model.encoder.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    mismatch = [k for k in set(expected) & set(state)
                if not torch.is_tensor(state[k]) or tuple(state[k].shape) != tuple(expected[k].shape)]
    if missing or extra or mismatch:
        raise ValueError(f"SSL encoder strict matching failed: missing={missing[:12]}, unexpected={extra[:12]}, shape={mismatch[:12]}; check the timm version and weight source")
    for key, tensor in state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"SSL weights contain non-finite values: {key}")
    model.encoder.load_state_dict(state, strict=True)
    result = {"path": str(Path(path).resolve()), "sha256": sha256(path), "format": kind,
              "loaded_encoder_tensors": len(state), "strict_match": True,
              "excluded_stage1_head_keys": unused,
              "note": "Tensor matching verifies structure, not historical training provenance."}
    if isinstance(raw, dict):
        result["source_epoch_if_stored"] = json_safe(raw.get("epoch"))
        result["source_best_score_if_stored"] = json_safe(raw.get("best_score"))
    return result


def ordinal_probabilities(logits):
    import torch
    q = torch.sigmoid(logits)
    p = torch.stack([1 - q[:, 0], torch.clamp(q[:, 0] - q[:, 1], min=1e-8),
                     torch.clamp(q[:, 1], min=1e-8)], dim=1)
    return p / p.sum(dim=1, keepdim=True)


def compute_loss(model, outputs, labels, cfg):
    import torch
    import torch.nn.functional as F
    main_logits, *rest = outputs
    aux_logits, features = rest[:4], rest[4]
    levels = torch.stack([(labels[:, 0] > k).float() for k in range(2)], dim=1)
    main_loss = F.binary_cross_entropy_with_logits(main_logits, levels, reduction="mean")
    aux_losses = []
    for i, logits in enumerate(aux_logits, 1):
        ce = F.cross_entropy(logits, labels[:, i], reduction="none")
        aux_losses.append(((1 - torch.exp(-ce)) ** cfg["focal_gamma"] * ce).mean())
    weighted = model.task_weighter([main_loss] + aux_losses)
    main_probs = ordinal_probabilities(main_logits)
    scores = torch.arange(3, dtype=main_probs.dtype, device=main_probs.device)
    main_score = (main_probs * scores).sum(dim=1)
    aux_score = torch.stack([(torch.softmax(x, dim=1) * scores).sum(dim=1)
                             for x in aux_logits], dim=1).mean(dim=1)
    consistency = F.smooth_l1_loss(main_score, aux_score, beta=1.0, reduction="mean")
    feat = F.normalize(features, p=2, dim=1, eps=1e-12)
    centers = F.normalize(model.main_centers, p=2, dim=1, eps=1e-12)[labels[:, 0]]
    center = ((feat - centers) ** 2).sum(dim=1).mean()
    total = weighted + cfg["consistency_weight"] * consistency + cfg["center_weight"] * center
    return total, main_probs


def augmentation_pipelines(cfg):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    crop = {"scale": (0.85, 1.0), "ratio": (0.75, 4 / 3), "interpolation": 1, "p": 1.0}
    if "size" in inspect.signature(A.RandomResizedCrop.__init__).parameters:
        crop["size"] = (224, 224)
    else:
        crop.update(height=224, width=224)
    if "num_holes_range" in inspect.signature(A.CoarseDropout.__init__).parameters:
        dropout = dict(num_holes_range=(1, 8), hole_height_range=(8, 16),
                       hole_width_range=(8, 16), fill=0, p=0.3)
    else:
        dropout = dict(min_holes=1, max_holes=8, min_height=8, max_height=16,
                       min_width=8, max_width=16, fill_value=0, p=0.3)
    train_ops = [A.RandomResizedCrop(**crop), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                 A.RandomRotate90(p=0.5), A.CoarseDropout(**dropout),
                 A.MotionBlur(blur_limit=(3, 5), allow_shifted=True, p=0.15),
                 A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1),
                                             brightness_by_max=True, p=0.25),
                 A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0), ToTensorV2()]
    val_ops = [A.Resize(height=224, width=224, interpolation=1, p=1.0),
               A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0), ToTensorV2()]
    seed_kwargs = {"seed": cfg["seed"]} if "seed" in inspect.signature(A.Compose.__init__).parameters else {}
    return A.Compose(train_ops, **seed_kwargs), A.Compose(val_ops, **seed_kwargs)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed); np.random.seed(seed)
    worker = torch.utils.data.get_worker_info()
    if worker and hasattr(worker.dataset.transform, "set_random_seed"):
        worker.dataset.transform.set_random_seed(seed)


class ImageDataset:
    """Map-style dataset; no skipped records or guessed labels."""
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image
        record = self.records[index]
        try:
            with Image.open(record["path"]) as source:
                rgb = np.array(source.convert("RGB"))
        except Exception as ex:
            raise RuntimeError(f"Unable to read image; sample not skipped: {record['filename']}") from ex
        return self.transform(image=rgb)["image"].float(), torch.tensor(record["labels"], dtype=torch.long)


def lr_factor(epoch_index, warmup_epochs, total_epochs):
    if epoch_index < warmup_epochs:
        return float(epoch_index + 1) / max(1, warmup_epochs)
    progress = float(epoch_index - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_encoder_frozen(model, frozen):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not frozen)


def confusion_metrics(cm):
    n = sum(sum(row) for row in cm)
    if n == 0:
        raise ValueError("Empty epoch")
    f1 = []
    for k in range(3):
        support = sum(cm[k]); predicted = sum(row[k] for row in cm)
        denom = support + predicted
        f1.append(2 * cm[k][k] / denom if denom else 0.0)
    return sum(cm[k][k] for k in range(3)) / n, sum(f1) / 3


def run_epoch(model, loader, optimizer, device, cfg, training, frozen=False):
    import torch
    model.train(training)
    if training and frozen:
        model.encoder.eval()  
    cm = torch.zeros((3, 3), dtype=torch.long, device=device)
    n = 0; summed_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, probabilities = compute_loss(model, model(images), labels, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss contains NaN/Inf; stopping with logs retained")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
        predictions = probabilities.argmax(dim=1)
        cm += torch.bincount(labels[:, 0] * 3 + predictions, minlength=9).reshape(3, 3)
        count = len(images); n += count; summed_loss += float(loss.detach()) * count
    if n != len(loader.dataset):
        raise RuntimeError(f"Processed {n} images, but the dataset contains {len(loader.dataset)}")
    accuracy, macro_f1 = confusion_metrics(cm.cpu().tolist())
    return {"images": n, "loss": summed_loss / n, "accuracy": accuracy, "macro_f1": macro_f1}


def environment_info():
    packages = {}
    for name in ["torch", "torchvision", "timm", "numpy", "pandas", "Pillow", "albumentations",
                 "opencv-python", "opencv-python-headless", "safetensors", "scikit-learn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages,
              "note": "Environment of this new execution, not proof of the historical environment."}
    if packages["torch"]:
        import torch
        result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    return result


def create_run_dir(root, prefix):
    directory = Path(root).expanduser().resolve() / (prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def train_model(cfg, train_records, val_records, input_report):
    import torch
    from torch.utils.data import DataLoader
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    directory = create_run_dir(cfg["output_root"], "reconstructed_train")
    logger = logging.getLogger("endoqual_reconstructed")
    logger.setLevel(logging.INFO); logger.propagate = False
    logger.handlers.clear()
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "training.log", encoding="utf-8")]:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")); logger.addHandler(handler)
    save_json(directory / "config.json", cfg)
    save_json(directory / "input_check.json", input_report)
    provenance = {"implementation_status": STATUS, "supplied_sources": SOURCES,
                  "reconstructed_script_sha256": sha256(__file__),
                  "historical_run_reproduced": False,
                  "manuscript_derived_unverified_settings": {"freeze_encoder_epochs": cfg['freeze_encoder_epochs'],
                                                             "early_stopping_patience": cfg['early_stopping_patience']},
                  "baseline_derived_unverified_settings": ["learning rates", "loss coefficients", "augmentation", "drop path"],
                  "new_implementation_choices": ["strict checkpoint matching", "zero workers by default", "disabled TF32",
                                                  "frozen encoder in eval mode", "explicit augmentation defaults", "complete logging"],
                  "selection_uses_test_data": False}
    save_json(directory / "provenance.json", provenance)
    save_json(directory / "environment.json", environment_info())
    try:
        logger.info("%s; new training output: %s", STATUS, directory)
        model = build_model(cfg)
        loading = load_ssl(model, cfg["ssl_checkpoint"])
        save_json(directory / "ssl_loading.json", loading)
        logger.info("SSL encoder strictly loaded: %s; SHA256=%s", loading['path'], loading['sha256'])
        model = model.to(device)
        train_transform, val_transform = augmentation_pipelines(cfg)
        save_json(directory / "transforms.json", {"training": train_transform.to_dict(), "validation": val_transform.to_dict(),
                                                  "stochastic_implementations_may_differ_between_package_versions": True})
        loaders = {}
        for offset, (name, records, transform) in enumerate([
                ("train", train_records, train_transform), ("val", val_records, val_transform)]):
            generator = torch.Generator().manual_seed(cfg["seed"] + offset)
            loaders[name] = DataLoader(ImageDataset(records, transform), batch_size=cfg["batch_size"],
                                       shuffle=name == "train", num_workers=cfg["num_workers"],
                                       pin_memory=device.type == "cuda", drop_last=False,
                                       worker_init_fn=seed_worker, generator=generator)
        parameter_groups = [{"params": model.encoder.parameters(), "lr": cfg["encoder_lr"], "name": "encoder"}]
        for name in ["main_head", "clarity_head", "wash_head", "blur_head", "brightness_head", "task_weighter"]:
            parameter_groups.append({"params": getattr(model, name).parameters(), "lr": cfg["heads_lr"], "name": name})
        parameter_groups.append({"params": [model.main_centers], "lr": cfg["center_lr"], "name": "centers"})
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=cfg["weight_decay"],
                                       betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"], amsgrad=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_factor(e, cfg["warmup_epochs"], cfg["epochs"]))
        history = []; best = -float("inf"); best_epoch = None; no_improvement = 0
        for epoch in range(1, cfg["epochs"] + 1):
            frozen = epoch <= cfg["freeze_encoder_epochs"]
            set_encoder_frozen(model, frozen)
            encoder_lr = optimizer.param_groups[0]["lr"]; heads_lr = optimizer.param_groups[1]["lr"]
            logger.info("Epoch %d/%d; encoder_frozen=%s; encoder_lr=%.8g; heads_lr=%.8g", epoch, cfg['epochs'], frozen, encoder_lr, heads_lr)
            training = run_epoch(model, loaders["train"], optimizer, device, cfg, True, frozen)
            validation = run_epoch(model, loaders["val"], optimizer, device, cfg, False)
            row = {"epoch": epoch, "encoder_frozen": frozen, "encoder_lr": encoder_lr, "heads_lr": heads_lr,
                   **{"train_" + k: v for k, v in training.items()}, **{"val_" + k: v for k, v in validation.items()}}
            history.append(row)
            with open(directory / "training_history.csv", "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            improved = validation["macro_f1"] > best
            if improved:
                best = validation["macro_f1"]; best_epoch = epoch; no_improvement = 0
            else:
                no_improvement += 1
            scheduler.step()
            
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            payload = {"epoch": epoch, "model_state_dict": cpu_state,
                       "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                       "history": list(history), "best_epoch": best_epoch, "best_val_macro_f1": best,
                       "epochs_no_improve": no_improvement, "config": cfg, "implementation_status": STATUS,
                       "provenance": provenance, "ssl_initialization": loading,
                       "rng_note": "This checkpoint is an audit record; exact resume state is not implemented."}
            if improved:
                torch.save(cpu_state, directory / "best_model_reconstructed.pth")
                torch.save(payload, directory / "best_checkpoint_reconstructed.pth")
            torch.save(payload, directory / "last_checkpoint_reconstructed.pth")
            del cpu_state, payload
            logger.info("train loss=%.6f; val loss=%.6f; val accuracy=%.6f; val macro-F1=%.6f; best_epoch=%s",
                        training['loss'], validation['loss'], validation['accuracy'], validation['macro_f1'], best_epoch)
            if no_improvement >= cfg["early_stopping_patience"]:
                logger.info("Early stopping after %d non-improving epochs", no_improvement)
                break
        best_path = directory / "best_model_reconstructed.pth"
        save_json(directory / "run_summary.json", {"status": "NEW_RECONSTRUCTED_TRAINING_COMPLETED",
                  "implementation_status": STATUS, "best_epoch": best_epoch, "best_val_macro_f1": best,
                  "epochs_executed": len(history), "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256(best_path),
                  "original_final_checkpoint_modified": False, "test_set_evaluated": False})
        logger.info("Completed. Original weight unchanged. New best model: %s", best_path)
        return directory
    except Exception:
        logger.exception("Execution failed; no claim of completed reproduction.")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close(); logger.removeHandler(handler)


def inspect_checkpoint(path, output_root):
    import torch
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = load_checkpoint(path)
    state = extract_state_dict(raw)
    tensors = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "elements": v.numel()}
               for k, v in state.items() if torch.is_tensor(v)}
    report = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path),
              "tensor_count": len(tensors), "tensors": tensors,
              "has_encoder_prefix": any(k.startswith('encoder.') for k in state),
              "has_five_mtl_heads": all(any(k.startswith(h + '.') for k in state)
                 for h in ['main_head', 'clarity_head', 'wash_head', 'blur_head', 'brightness_head']),
              "metadata": {k: json_safe(raw[k]) for k in ["epoch", "best_score", "best_epoch", "best_val_macro_f1",
                            "history", "config", "implementation_status"] if k in raw},
              "note": "Missing historical logs/epochs are not inferred from filenames or tensor values."}
    directory = create_run_dir(output_root, "checkpoint_inspection")
    save_json(directory / "checkpoint_inspection.json", report)
    save_json(directory / "inspection_environment.json", environment_info())
    print("Checkpoint inspection completed: ", directory / "checkpoint_inspection.json")
    print("Five multitask output heads present: ", report['has_five_mtl_heads'], "; history present: ", 'history' in report['metadata'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["check-inputs", "train", "inspect-checkpoint"], default="check-inputs")
    parser.add_argument("--config", type=Path, help="Optional JSON overriding recognized CONFIG entries")
    parser.add_argument("--csv", dest="annotations_csv")
    parser.add_argument("--data-root")
    parser.add_argument("--ssl-checkpoint")
    parser.add_argument("--output-root")
    parser.add_argument("--checkpoint", help="Checkpoint file for inspect-checkpoint mode")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--freeze-epochs", dest="freeze_encoder_epochs", type=int)
    parser.add_argument("--patience", dest="early_stopping_patience", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    cfg = copy.deepcopy(CONFIG)
    if args.config:
        override = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not isinstance(override, dict) or set(override) - set(cfg):
            raise ValueError("Configuration must be a JSON object containing recognized settings only")
        for key, value in override.items():
            if key == 'columns':
                if not isinstance(value, dict) or set(value) - set(cfg['columns']):
                    raise ValueError("Unrecognized field in columns configuration")
                cfg[key].update(value)
            else:
                cfg[key] = value
    for key in ["annotations_csv", "data_root", "ssl_checkpoint", "output_root", "device", "batch_size", "epochs",
                "freeze_encoder_epochs", "early_stopping_patience", "num_workers"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    print("Implementation status: ", STATUS)
    if args.mode == "inspect-checkpoint":
        if not args.checkpoint:
            parser.error("inspect-checkpoint requires --checkpoint")
        inspect_checkpoint(args.checkpoint, cfg["output_root"])
        return
    train, val, report = check_inputs(cfg)
    if args.mode == "check-inputs":
        directory = create_run_dir(cfg["output_root"], "input_check")
        save_json(directory / "input_check.json", report)
        save_json(directory / "config.json", cfg)
        print(json.dumps(report['groups'], ensure_ascii=False, indent=2))
        print("CSV and path checks passed; no weights loaded or training performed. Report: ", directory / "input_check.json")
    else:
        train_model(cfg, train, val, report)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as error:
        print("Stopped: ", error, file=sys.stderr)
        sys.exit(1)
