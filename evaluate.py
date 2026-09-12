#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EndoQual-MTL inference and statistical evaluation.
Exports predictions, classification metrics, calibration and clustered intervals.
Use --config configs/evaluate.json; see README.md for data and decoding rules."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG = {
    "checkpoint": "weights/endoqual_final.pth",
    "output_root": "runs/evaluation",
    "batch_size": 8,
    "num_workers": 0,              
    "device": "auto",             
    "seed": 42,
    "n_bootstrap": 1000,
    "calibration_bins": 5,         
    "min_valid_bootstrap_fraction": 0.95,
    "make_plots": True,
    "write_excel": True,
    "columns": {
        "filename": "filename",
        "patient_id": "patient_id",   
        "lesion_id": "lesion_id",     
        "main": "main_class",
        "structure_clarity": "structure_clarity",
        "stain_wash": "stain_wash",
        "focus": "blurriness",        
        "brightness": "brightness",
    },
    "cohorts": [
        {"name": "Internal", "csv": "data/annotations/internal_test.csv", "image_root": "data/internal",
         "split_column": "split", "split_value": "test"},
        {"name": "External", "csv": "data/annotations/external_test.csv", "image_root": "data/external",
         "split_column": None, "split_value": None},
    ],
    
    
    "distribution_only": [
        
        
    ],
}


VERSION = "1.0.0"
TASKS = ["main", "structure_clarity", "stain_wash", "focus", "brightness"]
DISPLAY = {"main": "Overall quality", "structure_clarity": "Structure clarity",
           "stain_wash": "Stain wash", "focus": "Focus (blurriness labels)", "brightness": "Brightness"}
CLASS_NAMES = {t: (["Poor", "Fair", "Good"] if t == "main" else ["Score 0", "Score 1", "Score 2"]) for t in TASKS}
MISSING = {"", "nan", "none", "null", "na", "n/a"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_csv_strings(path):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode CSV: {path}")


def parse_label(value, main=False):
    s = str(value).strip()
    if s.lower() in MISSING:
        return np.nan
    mapping = {"\u5dee": 0, "\u4e2d": 1, "\u597d": 2, "poor": 0, "fair": 1, "medium": 1, "good": 2,
               "poor quality": 0, "fair quality": 1, "medium quality": 1, "good quality": 2}
    if main and s.lower() in mapping:
        return mapping[s.lower()]
    try:
        number = float(s)
    except ValueError as ex:
        raise ValueError(f"Unrecognized label {value!r}; expected 0/1/2 or supported main-class names") from ex
    if not np.isfinite(number) or number not in (0., 1., 2.):
        raise ValueError(f"Invalid label {value!r}; labels must be integer grades without truncation or reversal")
    return int(number)


def load_annotations(spec, columns=None):
    cols = {**CONFIG["columns"], **(columns or {}), **spec.get("columns", {})}
    raw = read_csv_strings(spec["csv"])
    if spec.get("split_column"):
        col = spec["split_column"]
        if col not in raw:
            raise ValueError(f"{spec['csv']} has no split column {col!r}; specify the correct split configuration")
        raw = raw.loc[raw[col].str.strip() == str(spec["split_value"]).strip()].copy()
    if raw.empty:
        raise ValueError(f"{spec['name']} has no rows after filtering; check the CSV and split value")
    if cols["filename"] not in raw:
        raise ValueError(f"Missing filename column {cols['filename']!r}")
    out = pd.DataFrame(index=raw.index)
    out["filename"] = raw[cols["filename"]].str.strip().str.replace("\\", "/", regex=False)
    if out.filename.eq("").any() or out.filename.duplicated().any():
        raise ValueError(f"{spec['name']} contains empty or duplicate filenames; resolve these before evaluation")
    for key in ("patient_id", "lesion_id"):
        out[key] = raw[cols[key]].str.strip() if cols[key] in raw else ""
        out[key] = out[key].map(lambda s: "" if s.lower() in MISSING else s)
    for task in TASKS:
        source = raw[cols[task]] if cols[task] in raw else pd.Series("", index=raw.index)
        values = []
        for idx, value in source.items():
            try:
                values.append(parse_label(value, main=(task == "main")))
            except ValueError as ex:
                raise ValueError(f"{spec['name']} CSV row {idx + 2}, {cols[task]}: {ex}") from ex
        out[task + "_true"] = values
    return out.reset_index(drop=True)


def legacy_main_probabilities(cumulative):
    """Decode cumulative probabilities using subtraction, 1e-8 clipping and normalization.
    Clipping does not constrain the ordering of the raw cumulative outputs.
    """
    q = np.asarray(cumulative)
    if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("The main task requires two valid cumulative-probability columns")
    p = np.stack([1.0 - q[:, 0], np.maximum(q[:, 0] - q[:, 1], 1e-8),
                  np.maximum(q[:, 1], 1e-8)], axis=1)
    return p / p.sum(axis=1, keepdims=True)


def sampling_codes(frame, unit):
    if len(frame) == 0:
        raise ValueError("Resampling requires samples with reference labels")
    if unit == "image":
        return np.arange(len(frame)), len(frame)
    if unit not in ("patient", "lesion"):
        raise ValueError(f"Unknown resampling unit: {unit}")
    required = ["patient_id"] + (["lesion_id"] if unit == "lesion" else [])
    for col in required:
        if col not in frame or frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Complete {col} values are required for {unit}-clustered intervals")
    
    keys = pd.MultiIndex.from_frame(frame[required]) if unit == "lesion" else frame["patient_id"]
    codes, uniques = pd.factorize(keys, sort=False)
    if len(uniques) < 2:
        raise ValueError(f"Only {len(uniques)} {unit} clusters are available; clustered intervals cannot be estimated reliably")
    return codes.astype(int), len(uniques)


def weights_from_draws(codes, draws, n_clusters):
    return np.bincount(draws, minlength=n_clusters)[codes].astype(float)


def divide(a, b):
    return float(a / b) if b > 0 else np.nan


class Evaluator:
    """Cache ranks and bins; use cluster multiplicities as image weights during resampling."""
    def __init__(self, labels, probabilities, n_bins=5):
        y = np.asarray(labels)
        p = np.asarray(probabilities, dtype=float)
        if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all() or not np.isin(y, [0,1,2]).all():
            raise ValueError("Reference labels must be a nonempty array containing 0/1/2")
        if p.shape != (len(y), 3) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError("Probabilities must have shape N x 3 with all entries in [0,1]")
        if not np.allclose(p.sum(axis=1), 1., atol=2e-6, rtol=0):
            raise ValueError("Class probabilities do not sum to one; check main decoding or auxiliary softmax upstream")
        self.y, self.p = y.astype(int), p
        self.pred = p.argmax(axis=1)
        self.n = len(y)
        self.n_bins = int(n_bins)
        if self.n_bins < 2:
            raise ValueError("calibration_bins must be at least 2")
        self.onehot = np.eye(3)[self.y]
        self.brier = ((self.p - self.onehot)**2).sum(axis=1)
        self.nll = -np.log(np.clip(self.p[np.arange(self.n), self.y], 1e-15, 1.))
        self.targets = ["class_0", "class_1", "class_2", "top_label"]
        self.bin_indices, self.bin_prob, self.bin_event = [], [], []
        for k in range(4):
            prob = p[:, k] if k < 3 else p.max(axis=1)
            event = (self.y == k).astype(float) if k < 3 else (self.y == self.pred).astype(float)
            bins = np.minimum((prob * self.n_bins).astype(int), self.n_bins - 1)
            self.bin_indices.append(bins)
            self.bin_prob.append(prob)
            self.bin_event.append(event)
        self.auc_cache = []
        for k in range(3):
            order = np.argsort(p[:, k], kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(p[order, k]) != 0) + 1]
            self.auc_cache.append((order, starts, (self.y[order] == k)))

    def calibration_arrays(self, weights):
        counts, predictions, events = [], [], []
        for b, p, y in zip(self.bin_indices, self.bin_prob, self.bin_event):
            counts.append(np.bincount(b, weights=weights, minlength=self.n_bins))
            predictions.append(np.bincount(b, weights=weights*p, minlength=self.n_bins))
            events.append(np.bincount(b, weights=weights*y, minlength=self.n_bins))
        return np.array(counts), np.array(predictions), np.array(events)

    def metrics(self, weights=None):
        w = np.ones(self.n) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (self.n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid resampling weights")
        total = w.sum()
        cm = np.bincount(3*self.y + self.pred, weights=w, minlength=9).reshape(3,3)
        tp = cm.diagonal(); support = cm.sum(axis=1); predicted = cm.sum(axis=0)
        fn = support-tp; fp = predicted-tp; tn = total-tp-fn-fp
        m = {"overall_accuracy": float(tp.sum()/total)}
        f1s, sensitivities, aucs = [], [], []
        for k in range(3):
            sens = divide(tp[k], support[k]); f1 = divide(2*tp[k], 2*tp[k]+fp[k]+fn[k])
            order, starts, positive = self.auc_cache[k]
            pos = np.add.reduceat(w[order] * positive, starts)
            neg = np.add.reduceat(w[order] * ~positive, starts)
            
            auc = divide(np.sum(pos * (np.cumsum(neg) - .5*neg)), pos.sum()*neg.sum())
            aucs.append(auc); f1s.append(f1); sensitivities.append(sens)
            vals = {"sensitivity": sens, "specificity": divide(tn[k], tn[k]+fp[k]),
                    "accuracy_ovr": (tp[k]+tn[k])/total, "ppv": divide(tp[k], predicted[k]),
                    "npv": divide(tn[k], tn[k]+fn[k]), "f1": f1, "auc": auc,
                    "brier_ovr": np.average((self.p[:,k]-self.onehot[:,k])**2, weights=w)}
            m.update({f"class_{k}_{key}": float(value) for key, value in vals.items()})
        all_classes = bool(np.all(support > 0))
        m["macro_f1"] = float(np.mean(f1s)) if all_classes else np.nan
        m["balanced_accuracy"] = float(np.mean(sensitivities)) if all_classes else np.nan
        m["macro_auc_ovr"] = float(np.mean(aucs)) if all_classes else np.nan
        m["multiclass_brier"] = float(np.average(self.brier, weights=w))  
        m["log_loss"] = float(np.average(self.nll, weights=w))
        counts, probability_sums, event_sums = self.calibration_arrays(w)
        ece = np.abs(event_sums-probability_sums).sum(axis=1)/total
        for k in range(3):
            m[f"class_{k}_ece"] = float(ece[k])
        m["macro_classwise_ece"] = float(ece[:3].mean())
        m["top_label_ece"] = float(ece[3])
        return m

    def calibration_rows(self):
        counts, ps, ys = self.calibration_arrays(np.ones(self.n))
        rows = []
        for j, target in enumerate(self.targets):
            for b in range(self.n_bins):
                rows.append({"target": target, "bin": b, "bin_lower": b/self.n_bins,
                             "bin_upper": (b+1)/self.n_bins, "n_images": int(counts[j,b]),
                             "mean_probability": divide(ps[j,b],counts[j,b]),
                             "observed_fraction": divide(ys[j,b],counts[j,b])})
        return rows


def interval(values, total_replicates, min_fraction):
    finite = np.asarray(values)[np.isfinite(values)]
    required = max(2, int(np.ceil(total_replicates * min_fraction)))
    if len(finite) < required:
        return np.nan, np.nan, len(finite), "insufficient_valid_replicates"
    low, high = np.quantile(finite, [.025, .975])
    return float(low), float(high), len(finite), "ok"


def bootstrap_evaluate(evaluator, frame, unit, n_bootstrap, seed, min_fraction):
    point = evaluator.metrics(); keys = list(point)
    calibration = evaluator.calibration_rows()
    reason = None; n_clusters = 0
    try:
        codes, n_clusters = sampling_codes(frame, unit)
    except ValueError as ex:
        reason = str(ex)
    stats = np.full((n_bootstrap, len(keys)), np.nan)
    frequencies = np.full((n_bootstrap, 4, evaluator.n_bins), np.nan)
    if reason is None:
        rng = np.random.default_rng(seed)
        for b in range(n_bootstrap):
            draws = rng.integers(0, n_clusters, size=n_clusters)
            w = weights_from_draws(codes, draws, n_clusters)
            values = evaluator.metrics(w)
            stats[b] = [values[k] for k in keys]
            count, _, events = evaluator.calibration_arrays(w)
            np.divide(events, count, out=frequencies[b], where=count > 0)
    result = []
    for j, key in enumerate(keys):
        lo, hi, valid, status = interval(stats[:,j], n_bootstrap, min_fraction)
        if not np.isfinite(point[key]):
            lo = hi = np.nan; status = "undefined_point_estimate"
        result.append({"metric": key, "estimate": point[key], "ci_low": lo, "ci_high": hi,
                       "resampling_unit": unit, "n_clusters": n_clusters,
                       "valid_bootstraps": valid, "requested_bootstraps": n_bootstrap,
                       "ci_status": reason or status})
    for row in calibration:
        j = evaluator.targets.index(row["target"]); b = row["bin"]
        lo, hi, valid, status = interval(frequencies[:,j,b], n_bootstrap, min_fraction)
        mask = evaluator.bin_indices[j] == b
        subset = frame.loc[mask]
        row.update({"resampling_unit": unit, "ci_low": lo, "ci_high": hi,
                    "valid_bootstraps": valid, "ci_status": reason or status,
                    "n_patients": subset.patient_id.replace("", np.nan).nunique(),
                    "n_lesions": len(subset.loc[subset.patient_id.ne("") & subset.lesion_id.ne(""),
                                                     ["patient_id", "lesion_id"]].drop_duplicates())})
    return result, calibration


class EndoDataset:
    
    def __init__(self, frame, image_root, transform):
        self.names = frame.filename.tolist()
        self.root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        from PIL import Image
        path = self.root / self.names[idx]
        try:
            with Image.open(path) as im:
                image = np.array(im.convert("RGB"))
        except Exception as ex:
            raise RuntimeError(f"Unable to read image: {path}; evaluation stopped without dropping the sample") from ex
        return self.transform(image=image)["image"], idx


def build_model(checkpoint, device):
    import torch
    import torch.nn as nn
    import timm

    class DynamicTaskWeighting(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_vars = nn.Parameter(torch.zeros(5))

        def forward(self, losses):
            return sum(torch.exp(-self.log_vars[i])*loss+self.log_vars[i] for i,loss in enumerate(losses))

    class MultiTaskSwinModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model("swin_large_patch4_window7_224.ms_in22k",
                                             pretrained=False, num_classes=0, drop_path_rate=.3)
            d = self.encoder.num_features
            self.main_head = nn.Linear(d, 2)
            self.clarity_head = nn.Linear(d, 3)
            self.wash_head = nn.Linear(d, 3)
            self.blur_head = nn.Linear(d, 3)
            self.brightness_head = nn.Linear(d, 3)
            self.task_weighter = DynamicTaskWeighting()
            self.main_centers = nn.Parameter(torch.zeros(3, d))

        def forward(self, x):
            f = self.encoder(x)
            return self.main_head(f), self.clarity_head(f), self.wash_head(f), self.blur_head(f), self.brightness_head(f)

    model = MultiTaskSwinModel()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a recognized complete state_dict")
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    
    return model.float().to(device).eval()


def environment_info():
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "torch", "torchvision", "timm",
                 "albumentations", "opencv-python-headless", "opencv-python", "Pillow", "openpyxl", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def infer_all(config, run_dir):
    import os
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    try:
        import torch
        from torch.utils.data import DataLoader
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as ex:
        raise RuntimeError("Inference requires the PyTorch/timm/Albumentations environment described in README.md.") from ex
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found; update CONFIG['checkpoint']: {checkpoint}")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    selected_device = config["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if selected_device == "auto" else selected_device)
    frames = []
    for spec in config["cohorts"]:
        frame = load_annotations(spec, config["columns"])
        missing = [name for name in frame.filename if not (Path(spec["image_root"])/name).is_file()]
        if missing:
            raise FileNotFoundError(f"{spec['name']}: {len(missing)} images not found; examples: {missing[:5]}")
        frames.append(frame)
        print(f"{spec['name']}: {len(frame)} images; missing auxiliary labels: " +
              str({t:int(frame[t+'_true'].isna().sum()) for t in TASKS[1:]}), flush=True)
    print(f"Loading the same complete checkpoint; device={device}, FP32, batch_size={config['batch_size']}", flush=True)
    model = build_model(checkpoint, device)
    transform = A.Compose([A.Resize(224,224,interpolation=1),
                           A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]), ToTensorV2()])
    manifest = {"script_version": VERSION, "script_sha256": sha256(__file__),
                "checkpoint_sha256": sha256(checkpoint), "checkpoint_path": str(checkpoint.resolve()),
                "environment": environment_info(), "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cuda_version": torch.version.cuda, "precision": "float32; no autocast", "cohorts": []}
    for spec, frame in zip(config["cohorts"], frames):
        destination = run_dir / spec["name"]
        destination.mkdir()
        loader = DataLoader(EndoDataset(frame, spec["image_root"], transform),
                            batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=(device.type == "cuda"))
        chunks = []; done = 0
        with torch.inference_mode():
            for images, indices in loader:
                outputs = model(images.to(device, non_blocking=True))
                indices = indices.numpy()
                batch = frame.iloc[indices].copy()
                for j, task in enumerate(TASKS):
                    expected_width = 2 if j == 0 else 3
                    if outputs[j].shape != (len(indices), expected_width):
                        raise ValueError(f"{task} output shape does not match the model architecture")
                    logits = outputs[j].cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise ValueError(f"{task} model output contains NaN/Inf")
                    if j == 0:
                        cumulative = torch.sigmoid(outputs[j])
                        
                        values = torch.stack([1.0-cumulative[:,0],
                                              torch.clamp(cumulative[:,0]-cumulative[:,1],min=1e-8),
                                              torch.clamp(cumulative[:,1],min=1e-8)],dim=1)
                        probabilities = (values/values.sum(dim=1,keepdim=True)).cpu().numpy()
                        q = cumulative.cpu().numpy()
                        batch["main_q_gt0"] = q[:,0]; batch["main_q_gt1"] = q[:,1]
                        batch["main_order_violation"] = (q[:,1] > q[:,0]).astype(int)
                        batch["main_middle_clipped"] = (q[:,0]-q[:,1] < 1e-8).astype(int)
                    else:
                        probabilities = torch.softmax(outputs[j], dim=1).cpu().numpy()
                    batch[task+"_pred"] = probabilities.argmax(axis=1)
                    for k in range(3):
                        batch[f"{task}_p{k}"] = probabilities[:,k]
                    for k in range(expected_width):
                        batch[f"{task}_logit{k}"] = logits[:,k]
                chunks.append(batch); done += len(batch)
                print(f"\r{spec['name']} inference {done}/{len(frame)}", end="", flush=True)
        print()
        results = pd.concat(chunks).reset_index(drop=True)
        if results.filename.tolist() != frame.filename.tolist():
            raise RuntimeError("Prediction order or count mismatch; outputs were not saved")
        prediction_path = destination/"predictions.csv"
        results.to_csv(prediction_path, index=False, encoding="utf-8-sig", float_format="%.17g")
        template = frame.rename(columns={t+"_true":config["columns"][t] for t in TASKS})
        template.to_csv(destination/"metadata_check_template.csv",index=False,encoding="utf-8-sig")
        manifest["cohorts"].append({"name": spec["name"], "n_images": len(frame),
                                     "annotation_sha256":sha256(spec["csv"]),
                                     "predictions_sha256":sha256(prediction_path),
                                     "order_violation_n":int(results.main_order_violation.sum()),
                                     "order_violation_fraction":float(results.main_order_violation.mean())})
        save_json(run_dir/"inference_manifest.json", manifest)
    return manifest


def load_predictions_for_analysis(spec, config, run_dir):
    path = run_dir/spec["name"]/"predictions.csv"
    pred = pd.read_csv(path, encoding="utf-8-sig", dtype={"filename":str,"patient_id":str,"lesion_id":str},
                       keep_default_na=False, float_precision="round_trip")
    ann = load_annotations(spec, config["columns"])
    if pred.filename.duplicated().any() or set(pred.filename) != set(ann.filename):
        raise ValueError(f"{spec['name']}: CSV samples do not match saved predictions; keep the evaluation sample set unchanged")
    old = pred.set_index("filename")
    for task in TASKS:
        col = task+"_true"
        if col in old:
            before = pd.to_numeric(ann.filename.map(old[col]),errors="coerce").to_numpy()
            after = ann[col].to_numpy(dtype=float)
            changed = np.isfinite(before) & (~np.isfinite(after) | (before != after))
            if changed.any():
                raise ValueError(f"{spec['name']} {task} reference labels were changed or removed; review the inputs and use a separate run when appropriate")
    
    remove = ["patient_id", "lesion_id"] + [t+"_true" for t in TASKS]
    result = ann.merge(pred.drop(columns=[c for c in remove if c in pred]), on="filename", how="left", validate="one_to_one")
    for task in TASKS:
        probcols = [f"{task}_p{k}" for k in range(3)]
        result[probcols] = result[probcols].apply(pd.to_numeric, errors="raise")
        reported_pred = pd.to_numeric(result[task+"_pred"],errors="raise").to_numpy()
        if not np.array_equal(reported_pred, result[probcols].to_numpy().argmax(axis=1)):
            raise ValueError(f"{spec['name']} {task} predicted labels do not match the probability argmax")
    return result, {"cohort":spec["name"], "prediction_sha256":sha256(path), "annotation_sha256":sha256(spec["csv"])}


def distributions(frame, cohort):
    rows = []
    for task in TASKS:
        label = frame[task+"_true"]
        n_labeled = int(label.notna().sum())
        for k in range(3):
            subset = frame.loc[label.eq(k)]
            rows.append({"cohort":cohort, "task":task, "class":CLASS_NAMES[task][k], "class_code":k,
                         "n_images":len(subset), "percent_of_labeled":100*len(subset)/n_labeled if n_labeled else np.nan,
                         "n_labeled":n_labeled, "n_missing_labels":len(frame)-n_labeled, "n_total_images":len(frame),
                         "n_patients_with_this_class":subset.patient_id.replace("",np.nan).nunique(),
                         "n_lesions_with_this_class":len(subset.loc[subset.patient_id.ne("") & subset.lesion_id.ne(""),
                                                                           ["patient_id","lesion_id"]].drop_duplicates())})
    return rows


def formatted_estimate(row):
    if not np.isfinite(row["estimate"]):
        return "NA (undefined)"
    if row["ci_status"] != "ok":
        return f"{row['estimate']:.4f} (CI unavailable)"
    return f"{row['estimate']:.4f} ({row['ci_low']:.4f}-{row['ci_high']:.4f})"


def presentation_table(metrics, tasks, selected_metrics=None):
    if metrics.empty:
        return pd.DataFrame()
    subset = metrics.loc[metrics.task.isin(tasks) & metrics.resampling_unit.isin(["patient","lesion"])].copy()
    if selected_metrics is not None:
        subset = subset.loc[subset.metric.isin(selected_metrics)]
    subset["value"] = subset.apply(formatted_estimate,axis=1)
    subset["column"] = subset.cohort + " / " + subset.resampling_unit + " bootstrap"
    return subset.pivot(index=["task","metric"],columns="column",values="value").reset_index()


def make_figures(cohort, task, e, metrics, calibration, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, confusion_matrix
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "savefig.dpi":300})
    directory.mkdir(exist_ok=True)
    available = metrics.loc[metrics.resampling_unit.eq("patient") & metrics.ci_status.eq("ok")]
    unit = "patient" if not available.empty else "image"
    selected = metrics.loc[metrics.resampling_unit.eq(unit)].set_index("metric")
    colors = ["#2369A1", "#D47B21", "#2D8A68"]
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    for k, name in enumerate(CLASS_NAMES[task]):
        if np.unique(e.y == k).size < 2:
            continue
        fpr,tpr,_ = roc_curve((e.y==k).astype(int),e.p[:,k])
        row = selected.loc[f"class_{k}_auc"]
        label = f"{name}: AUC {row.estimate:.3f}"
        if row.ci_status == "ok":
            label += f" ({row.ci_low:.3f}-{row.ci_high:.3f})"
        ax.plot(fpr,tpr,color=colors[k],label=label)
    ax.plot([0,1],[0,1],"--",color="0.65")
    ax.set(xlabel="False positive rate",ylabel="True positive rate",xlim=(0,1),ylim=(0,1),
           title=f"{cohort}: {DISPLAY[task]}\n95% CIs: {unit} bootstrap")
    ax.legend(loc="lower right",fontsize=8)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(directory/f"{cohort}_{task}_ROC.{ext}")
    plt.close(fig)
    cm = confusion_matrix(e.y,e.pred,labels=[0,1,2])
    pd.DataFrame(cm,index=CLASS_NAMES[task],columns=CLASS_NAMES[task]).to_csv(directory/f"{cohort}_{task}_confusion.csv")
    fig,ax = plt.subplots(figsize=(5.5,4.7))
    im = ax.imshow(cm,cmap="Blues"); fig.colorbar(im,ax=ax)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASS_NAMES[task],yticklabels=CLASS_NAMES[task],
           xlabel="Predicted label",ylabel="Expert reference label",title=f"{cohort}: {DISPLAY[task]}")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(directory/f"{cohort}_{task}_confusion.{ext}")
    plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(9,8))
    cal = calibration.loc[calibration.resampling_unit.eq(unit)]
    for j,(target,ax) in enumerate(zip(e.targets,axes.flat)):
        rows = cal.loc[cal.target.eq(target) & cal.n_images.gt(0)].sort_values("bin")
        ax.plot([0,1],[0,1],"--",color="0.6")
        ax.plot(rows.mean_probability,rows.observed_fraction,"o-",color=colors[j%3])
        for _,r in rows.iterrows():
            if r.ci_status == "ok":
                ax.vlines(r.mean_probability,r.ci_low,r.ci_high,color=colors[j%3],alpha=.75)
            ax.annotate(f"n={int(r.n_images)}",(r.mean_probability,r.observed_fraction),xytext=(2,6),textcoords="offset points",fontsize=7)
        ax.set(xlim=(-.03,1.03),ylim=(-.03,1.08),xlabel="Mean predicted probability",ylabel="Observed fraction",
               title=CLASS_NAMES[task][j] if j<3 else "Top-label confidence vs accuracy")
    fig.suptitle(f"{cohort}: {DISPLAY[task]}\nFixed equal-width bins; pointwise 95% CIs: {unit} bootstrap")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(directory/f"{cohort}_{task}_calibration.{ext}")
    plt.close(fig)


def write_methods(path, config, coverage):
    text = f"""ANALYSIS SETTINGS / REPRODUCIBLE METHODS NOTES
Script version: {VERSION}; seed: {config['seed']}; bootstrap replicates: {config['n_bootstrap']}.

Five tasks: overall quality, structure clarity, stain wash, focus (source column blurriness), brightness.
Overall reference labels: Poor=0, Fair=1, Good=2. Auxiliary labels are retained as 0/1/2.
Confirm the clinical direction of the original blurriness labels; this script does not reverse them.

Decoding preserves the supplied inference code. For the two primary logits z0,z1,
q0=sigmoid(z0), q1=sigmoid(z1), v=(1-q0, max(q0-q1,1e-8), max(q1,1e-8)).
Probabilities equal v/sum(v); the predicted class is argmax (first class in a tie).
This is NOT counting sigmoid outputs above 0.5. Sigmoid, primary probability conversion and auxiliary
softmax are computed in PyTorch float32 on the inference device, as in the supplied evaluation code.
Reproducibility still depends on the same checkpoint, data, preprocessing and software/hardware.
The raw frequency q1>q0 is reported. Clipping does not impose monotonicity on the neural network.
Each auxiliary head uses three-class softmax followed by argmax. Report results for these actual
outputs; do not describe the auxiliary heads as ordinal heads merely because labels are ordered.

All point estimates are IMAGE-WEIGHTED estimates on each task's labeled subset.
Image, patient, and lesion bootstraps change the resampling unit, not the point-estimate estimand.
Patients are resampled with replacement, retaining every image of each sampled patient each time.
Lesions use the composite (patient_id, lesion_id) key and retain all images each time sampled.
The patient bootstrap is primary because it captures dependence across lesions in a patient;
the separate lesion bootstrap is a sensitivity analysis and does not capture all patient dependence.
These intervals condition on the fitted model/reference labels and exclude model-retraining and
expert-annotation uncertainty. They are not patient-level diagnostic accuracy estimates.

All class metrics are one-versus-rest. Macro-F1 and macro-AUC are unweighted means over 3 classes.
Ratios with zero denominators and AUC with no positive/negative class are undefined (NA), not zero.
Macro metrics are undefined if a true class is absent. The 2.5th/97.5th percentiles use NumPy's
linear quantile interpolation, which differs slightly from selecting integer order statistics in
the old script. Invalid bootstrap estimates are excluded per metric and their counts are reported.
CIs are suppressed when fewer than {config['min_valid_bootstrap_fraction']:.0%} of replicates are valid.
Sparse-class intervals with excluded replicates require cautious interpretation.

Calibration: {config['calibration_bins']} fixed equal-width bins on [0,1]; intervals are left-closed/right-open
except the last includes 1. Classwise curves compare mean class probability against the observed
reference-class fraction. Top-label curves compare maximum probability against classification
correctness. Classwise ECE=sum_b n_b/N*abs(observed_b-predicted_b), ignoring empty bins; macro ECE
is the arithmetic mean of three classwise ECEs. Top-label ECE uses confidence/correctness bins.
Multiclass Brier=sum_k(p_k-onehot_k)^2 averaged over images, range [0,2], with NO division by 2 or 3.
Brier measures overall probability quality, not calibration alone. Log loss clips at 1e-15.
Reliability-curve CIs are pointwise clustered-bootstrap intervals for observed fractions in fixed
probability bins, not simultaneous bands. The x-coordinate is the original bin mean probability.
Sample and patient/lesion counts per bin are supplied; sparse bins may have no reliable interval.
No post-hoc calibrator is fitted. Test data must not be used to fit a calibration transformation.

Missing auxiliary labels produce clearly labeled partial/no evaluations. Missing cluster IDs never
fall back to pretending individual images are independent patients. All missingness is recorded.
Training/validation distributions are included only if configured. These notes document executed
analysis settings, not a claim that missing data or limited generalizability have been resolved.

COVERAGE:\n{coverage.to_string(index=False)}\n
Primary sources:
https://scikit-learn.org/stable/modules/calibration.html
https://proceedings.mlr.press/v70/guo17a.html
"""
    path.write_text(text,encoding="utf-8")


def analyze_all(config, run_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = run_dir/f"analysis_{stamp}"
    out.mkdir(parents=True)
    metric_rows, cal_rows, distribution_rows, coverage_rows, qc_rows, provenance = [], [], [], [], [], []
    plot_jobs = []
    complete = True
    for cohort_index, spec in enumerate(config["cohorts"]):
        frame, prov = load_predictions_for_analysis(spec,config,run_dir)
        provenance.append(prov)
        distribution_rows.extend(distributions(frame,spec["name"]))
        if "main_q_gt0" in frame and "main_q_gt1" in frame:
            q0 = pd.to_numeric(frame.main_q_gt0).to_numpy(); q1 = pd.to_numeric(frame.main_q_gt1).to_numpy()
            reconstructed = legacy_main_probabilities(np.column_stack([q0,q1]).astype(np.float32))
            saved = frame[[f"main_p{k}" for k in range(3)]].to_numpy()
            if not np.allclose(reconstructed,saved,atol=2e-6,rtol=0):
                raise ValueError("Saved main probabilities do not match the decoding rule; check the prediction files")
            qc_rows.append({"cohort":spec["name"],"check":"raw_cumulative_order_violation", "n":int((q1>q0).sum()),
                            "denominator":len(frame),"fraction":float((q1>q0).mean())})
        for task_index,task in enumerate(TASKS):
            subset = frame.loc[frame[task+"_true"].notna()].copy().reset_index(drop=True)
            issues = []
            for unit in ("patient","lesion"):
                try:
                    sampling_codes(subset,unit)
                except ValueError as ex:
                    issues.append(str(ex))
            if len(subset) != len(frame):
                issues.append(f"missing_reference_labels={len(frame)-len(subset)}")
            present_classes = int(subset[task+"_true"].nunique())
            if present_classes != 3:
                issues.append(f"only_{present_classes}_reference_classes")
            complete = complete and not issues
            coverage_rows.append({"cohort":spec["name"],"task":task,"n_total_images":len(frame),
                                  "n_evaluated_images":len(subset),"n_missing_labels":len(frame)-len(subset),
                                  "n_images_missing_patient_id":int(subset.patient_id.eq("").sum()),
                                  "n_images_missing_lesion_id":int(subset.lesion_id.eq("").sum()),
                                  "n_patients":subset.patient_id.replace("",np.nan).nunique(),
                                  "n_lesions":len(subset.loc[subset.patient_id.ne("") & subset.lesion_id.ne(""),
                                                                    ["patient_id","lesion_id"]].drop_duplicates()),
                                  "status":"complete_inputs" if not issues else "INCOMPLETE: " + "; ".join(issues)})
            if subset.empty:
                print(f"{spec['name']} {task}: no reference labels; retaining predictions and skipping performance calculations",flush=True)
                continue
            e = Evaluator(subset[task+"_true"].to_numpy(),subset[[f"{task}_p{k}" for k in range(3)]].to_numpy(),config["calibration_bins"])
            tm,tc = [],[]
            for unit_index,unit in enumerate(("image","patient","lesion")):
                print(f"{spec['name']} / {task} / {unit}: {config['n_bootstrap']} bootstrap",flush=True)
                
                seed = config["seed"] + cohort_index*1000 + unit_index*100
                mr,cr = bootstrap_evaluate(e,subset,unit,config["n_bootstrap"],seed,config["min_valid_bootstrap_fraction"])
                for row in mr+cr:
                    row.update({"cohort":spec["name"],"task":task,"n_evaluated_images":len(subset)})
                tm.extend(mr);tc.extend(cr)
            metric_rows.extend(tm);cal_rows.extend(tc)
            plot_jobs.append((spec["name"],task,e,pd.DataFrame(tm),pd.DataFrame(tc)))
    for spec in config.get("distribution_only",[]):
        distribution_rows.extend(distributions(load_annotations(spec,config["columns"]),spec["name"]))
    metrics = pd.DataFrame(metric_rows); calibration = pd.DataFrame(cal_rows)
    dist = pd.DataFrame(distribution_rows); coverage = pd.DataFrame(coverage_rows); qc = pd.DataFrame(qc_rows)
    core = ["overall_accuracy","macro_f1","macro_auc_ovr","multiclass_brier","macro_classwise_ece","top_label_ece"]
    primary_metrics = core + [f"class_{k}_{m}" for k in range(3) for m in ["sensitivity","specificity","ppv","npv","f1","auc","ece"]]
    primary = presentation_table(metrics,["main"],primary_metrics)
    auxiliary = presentation_table(metrics,TASKS[1:],core)
    frames = {"main_results_table":primary, "auxiliary_summary_table":auxiliary,
              "all_metrics_with_CI":metrics,"calibration_bins_with_CI":calibration,
              "class_distributions":dist,"coverage_check":coverage,"ordinal_QC":qc}
    for name,df in frames.items():
        df.to_csv(out/f"{name}.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    failed_primary_ci = 0 if metrics.empty else int(((metrics.resampling_unit.eq("patient")) & metrics.ci_status.ne("ok")).sum())
    ready = complete and failed_primary_ci == 0
    summary = {"all_required_test_inputs_present":bool(complete),"primary_patient_ci_unavailable_count":failed_primary_ci,
               "ready_for_author_review":bool(ready),"automatic_submission_ready":False,
               "note":"Verify label coding, cohort definitions, model version and input completeness before reporting.",
               "training_validation_distributions_included":len(config.get("distribution_only",[])),
               "environment":environment_info(),"sources":provenance,"config":config}
    save_json(out/"analysis_manifest.json",summary)
    write_methods(out/"methods_and_decoding.txt",config,coverage)
    if config.get("write_excel",True):
        try:
            with pd.ExcelWriter(out/"reviewer_2_3_results.xlsx",engine="openpyxl") as writer:
                for name,df in frames.items():
                    df.to_excel(writer,sheet_name=name[:31],index=False)
                    sheet=writer.sheets[name[:31]];sheet.freeze_panes="A2";sheet.auto_filter.ref=sheet.dimensions
                    for cells in sheet.columns:
                        sheet.column_dimensions[cells[0].column_letter].width=min(48,max(14,max(len(str(c.value or "")) for c in cells[:50])+2))
        except ImportError:
            print("openpyxl is unavailable; all CSV files were retained. Install openpyxl to also export Excel.",flush=True)
    if config.get("make_plots",True):
        for args in plot_jobs:
            make_figures(*args,out/"figures")
    if not complete:
        prefix = "INCOMPLETE_INPUTS_REVIEW_REQUIRED"
    elif not ready:
        prefix = "SOME_CIS_NOT_ESTIMABLE_REVIEW_REQUIRED"
    else:
        prefix = "INPUTS_COMPLETE_REVIEW_REQUIRED"
    (out/f"{prefix}.txt").write_text(summary["note"] + "\n\n" + coverage.to_string(index=False),encoding="utf-8")
    print(f"\nAnalysis output: {out.resolve()}\nStatus: {prefix}",flush=True)
    return out


def validate_config(config):
    if not isinstance(config["n_bootstrap"],int) or config["n_bootstrap"] < 2:
        raise ValueError("n_bootstrap must be at least 2; use 1000 or more for reporting")
    if not 0 < config["min_valid_bootstrap_fraction"] <= 1:
        raise ValueError("min_valid_bootstrap_fraction must be in (0,1]")
    names = [s["name"] for s in config["cohorts"]]
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+",x) for x in names):
        raise ValueError("Cohort names must be unique and contain only letters, digits, underscores or hyphens")


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage",choices=["all","infer","analyze"],default="all")
    parser.add_argument("--run-dir",help="Existing prediction directory; uses its saved config.json")
    parser.add_argument("--config",help="Optional complete JSON configuration overriding built-in or saved settings")
    args = parser.parse_args()
    config = copy.deepcopy(CONFIG)
    if args.stage == "analyze":
        if not args.run_dir:
            parser.error("--stage analyze requires --run-dir")
        run_dir = Path(args.run_dir)
        config = json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.stage in ("all","infer"):
        if args.run_dir:
            parser.error("Inference creates a new directory; --run-dir is only for reusing predictions")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path(config["output_root"])/f"run_{stamp}"
        run_dir.mkdir(parents=True)
        
        config["checkpoint"] = str(Path(config["checkpoint"]).resolve())
        for spec in config["cohorts"]+config.get("distribution_only",[]):
            spec["csv"] = str(Path(spec["csv"]).resolve())
            if "image_root" in spec:
                spec["image_root"] = str(Path(spec["image_root"]).resolve())
        save_json(run_dir/"config.json",config)
        print(f"Run directory: {run_dir.resolve()}",flush=True)
        infer_all(config,run_dir)
    if args.stage in ("all","analyze"):
        analyze_all(config,run_dir)
    print(f'After completing CSV labels/identifiers, reuse predictions with:\npython "{Path(__file__).name}" --stage analyze --run-dir "{run_dir.resolve()}"',flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEvaluation stopped: {exc}",file=sys.stderr)
        raise
