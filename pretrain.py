#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Degradation-aware self-supervised pretraining.
Adapted from retained source with configurable paths and recorded runs.
See README.md for configuration provenance and initialization requirements."""
import argparse
import importlib.metadata
import platform
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import os
import math
import copy
import random
import glob
import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm

from safetensors.torch import load_file as load_safetensors
import albumentations as A
from albumentations.pytorch import ToTensorV2


PACKAGE_NAMES = (
    'torch', 'torchvision', 'timm', 'numpy', 'pandas', 'Pillow',
    'albumentations', 'opencv-python', 'opencv-python-headless',
    'safetensors', 'tensorboard', 'tqdm', 'scikit-learn',
    'matplotlib', 'openpyxl',
)

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def collect_environment():
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'record_type': 'current_execution_environment',
        'python': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'operating_system': platform.system(),
        'architecture': platform.machine(),
        'packages': packages,
        'note': 'A current environment snapshot does not establish the historical training environment.',
    }
    try:
        import torch
        available = bool(torch.cuda.is_available())
        result['pytorch_runtime'] = {
            'torch_version': str(torch.__version__),
            'cuda_build_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'cuda_available': available,
            'gpus': [
                {
                    'index': i,
                    'model': torch.cuda.get_device_name(i),
                    'memory_bytes': int(torch.cuda.get_device_properties(i).total_memory),
                    'compute_capability': list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ] if available else [],
            'cuda_matmul_allow_tf32_at_export': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32_at_export': bool(torch.backends.cudnn.allow_tf32),
        }
    except Exception as exc:
        # Exception messages can contain personal installation paths.
        result['pytorch_runtime'] = {'inspection_status': 'unavailable', 'error_type': type(exc).__name__}
    return result

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_blur(img, severity):
    if severity == 1:
        k = 3
    elif severity == 2:
        k = 5
    else:
        k = 7
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_brightness_shift(img, severity):
    img = img.astype(np.float32)
    if severity == 1:
        alpha = random.choice([0.90, 1.10])
        beta = random.choice([-8, 8])
    elif severity == 2:
        alpha = random.choice([0.80, 1.20])
        beta = random.choice([-18, 18])
    else:
        alpha = random.choice([0.70, 1.30])
        beta = random.choice([-30, 30])
    out = img * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_stain_nonuniformity(img, severity):
    h, w, _ = img.shape
    img = img.astype(np.float32)

    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    cx = random.uniform(-0.4, 0.4)
    cy = random.uniform(-0.4, 0.4)
    sigma = random.uniform(0.4, 0.8)

    field = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)

    if severity == 1:
        amp = random.uniform(0.10, 0.18)
    elif severity == 2:
        amp = random.uniform(0.18, 0.28)
    else:
        amp = random.uniform(0.28, 0.40)

    modifier = 1.0 + amp * (field * 2 - 1)
    modifier = np.repeat(modifier[:, :, None], 3, axis=2)

    out = img * modifier
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_occlusion(img, severity):
    out = img.copy()
    h, w, _ = out.shape

    if severity == 1:
        num_holes, scale = 1, 0.08
    elif severity == 2:
        num_holes, scale = 2, 0.12
    else:
        num_holes, scale = 3, 0.16

    for _ in range(num_holes):
        hole_w = int(w * random.uniform(scale * 0.6, scale))
        hole_h = int(h * random.uniform(scale * 0.6, scale))
        x1 = random.randint(0, max(0, w - hole_w))
        y1 = random.randint(0, max(0, h - hole_h))

        if random.random() < 0.5:
            fill = random.randint(0, 30)
        else:
            fill = random.randint(220, 255)
        out[y1:y1 + hole_h, x1:x1 + hole_w] = fill

    return out


def apply_degradation(img, deg_type, severity):
    if deg_type == 0 or severity == 0:
        return img
    if deg_type == 1:
        return apply_blur(img, severity)
    if deg_type == 2:
        return apply_brightness_shift(img, severity)
    if deg_type == 3:
        return apply_stain_nonuniformity(img, severity)
    if deg_type == 4:
        return apply_occlusion(img, severity)
    return img


def sample_degradation():
    
    if random.random() < 0.20:
        return 0, 0
    deg_type = random.choice([1, 2, 3, 4])
    severity = random.choice([1, 2, 3])
    return deg_type, severity


class QualityRankingPretrainDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform
        print(f"Pretraining images: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as exc:
            raise RuntimeError(f"Cannot read image: {img_path}") from exc

        type_a, sev_a = sample_degradation()
        type_b, sev_b = sample_degradation()

        img_a = apply_degradation(image.copy(), type_a, sev_a)
        img_b = apply_degradation(image.copy(), type_b, sev_b)


        quality_a = 3 - sev_a
        quality_b = 3 - sev_b

        
        if quality_a == quality_b:
            if random.random() < 0.5:
                type_a, sev_a, img_a, quality_a, type_b, sev_b, img_b, quality_b = \
                    type_b, sev_b, img_b, quality_b, type_a, sev_a, img_a, quality_a

        rank_label = 1 if quality_a > quality_b else 0  

        if self.transform:
            img_a = self.transform(image=img_a)["image"]
            img_b = self.transform(image=img_b)["image"]

        return (
            img_a,
            img_b,
            torch.tensor(type_a, dtype=torch.long),
            torch.tensor(sev_a, dtype=torch.long),
            torch.tensor(type_b, dtype=torch.long),
            torch.tensor(sev_b, dtype=torch.long),
            torch.tensor(rank_label, dtype=torch.float32),
        )


class QualityAwarePretrainModel(nn.Module):
    def __init__(self, pretrained_path=None, device="cpu", drop_path_rate=0.3):
        super().__init__()

        self.encoder = timm.create_model(
            "swin_large_patch4_window7_224.ms_in22k",
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate
        )
        print(f"Model created: Swin-Large, drop-path rate: {drop_path_rate}")

        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading Swin Transformer weights from '{pretrained_path}'...")
            state_dict = load_safetensors(pretrained_path, device=str(device))
            
            expected = self.encoder.state_dict()
            missing = sorted(set(expected) - set(state_dict))
            extra = [k for k in state_dict if k not in expected and not k.startswith("head.")]
            bad_shape = [k for k in expected if k in state_dict and expected[k].shape != state_dict[k].shape]
            if missing or extra or bad_shape:
                raise ValueError(f"ImageNet encoder mismatch: missing={missing[:8]}, extra={extra[:8]}, shapes={bad_shape[:8]}")
            self.encoder.load_state_dict({k: state_dict[k] for k in expected}, strict=True)
            print("ImageNet weights loaded successfully.")
        else:
            raise FileNotFoundError(f"ImageNet initialization is required: {pretrained_path}")

        feat_dim = self.encoder.num_features
        self.deg_type_head = nn.Linear(feat_dim, 5)
        self.severity_head = nn.Linear(feat_dim, 4)
        self.quality_head = nn.Linear(feat_dim, 1)

    def forward_once(self, x):
        feat = self.encoder(x)
        deg_type_logits = self.deg_type_head(feat)
        severity_logits = self.severity_head(feat)
        quality_score = self.quality_head(feat).squeeze(1)
        return deg_type_logits, severity_logits, quality_score

    def forward(self, img_a, img_b):
        out_a = self.forward_once(img_a)
        out_b = self.forward_once(img_b)
        return out_a, out_b


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_pretrain(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    deg_correct = 0
    sev_correct = 0
    rank_correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="val"):
            if batch is None:
                continue

            img_a, img_b, type_a, sev_a, type_b, sev_b, rank_label = batch
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            type_a = type_a.to(device)
            sev_a = sev_a.to(device)
            type_b = type_b.to(device)
            sev_b = sev_b.to(device)
            rank_label = rank_label.to(device)

            (deg_a, sev_logits_a, score_a), (deg_b, sev_logits_b, score_b) = model(img_a, img_b)

            loss_deg = F.cross_entropy(deg_a, type_a) + F.cross_entropy(deg_b, type_b)
            loss_sev = F.cross_entropy(sev_logits_a, sev_a) + F.cross_entropy(sev_logits_b, sev_b)
            rank_logits = score_a - score_b
            loss_rank = F.binary_cross_entropy_with_logits(rank_logits, rank_label)

            loss = 1.0 * loss_deg + 1.0 * loss_sev + 0.5 * loss_rank

            bs = img_a.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            deg_pred_a = deg_a.argmax(dim=1)
            sev_pred_a = sev_logits_a.argmax(dim=1)
            rank_pred = (torch.sigmoid(rank_logits) > 0.5).long()

            deg_correct += (deg_pred_a == type_a).sum().item()
            sev_correct += (sev_pred_a == sev_a).sum().item()
            rank_correct += (rank_pred == rank_label.long()).sum().item()

    avg_loss = total_loss / max(1, total_samples)
    deg_acc = deg_correct / max(1, total_samples)
    sev_acc = sev_correct / max(1, total_samples)
    rank_acc = rank_correct / max(1, total_samples)

    score = 0.35 * deg_acc + 0.35 * sev_acc + 0.30 * rank_acc
    return avg_loss, deg_acc, sev_acc, rank_acc, score


def pretrain_quality_aware(cfg):
    set_seed(cfg["seed"])

    UNLABELED_DATA_DIR = cfg["unlabeled_image_root"]
    EXISTING_PRETRAINED_PATH = cfg["imagenet_checkpoint"]
    if not Path(EXISTING_PRETRAINED_PATH).is_file():
        raise FileNotFoundError(EXISTING_PRETRAINED_PATH)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    OUTPUT_DIR = str(Path(cfg["output_root"]) / ("pretrain_" + stamp))
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=False)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    FINAL_ENCODER_FILENAME = "encoder_best_export.pth"
    BEST_ENCODER_FILENAME = "encoder_best.pth"
    BEST_CHECKPOINT_FILENAME = "pretrain_best.pth"
    LAST_CHECKPOINT_FILENAME = "pretrain_last.pth"
    BATCH_SIZE = cfg["batch_size"]
    NUM_EPOCHS = cfg["epochs"]
    WARMUP_EPOCHS = cfg["warmup_epochs"]
    ENCODER_LR = cfg["encoder_lr"]
    HEADS_LR = cfg["heads_lr"]
    WEIGHT_DECAY = cfg["weight_decay"]
    DROP_PATH_RATE = cfg["drop_path_rate"]
    EARLY_STOPPING_PATIENCE = cfg["early_stopping_patience"]
    SAVE_EVERY_N_EPOCHS = cfg["save_every_n_epochs"]
    Path(OUTPUT_DIR, "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf8")
    Path(OUTPUT_DIR, "environment.json").write_text(json.dumps(collect_environment(), indent=2), encoding="utf8")
    Path(OUTPUT_DIR, "source_record.json").write_text(json.dumps({
        "script_sha256": file_hash(__file__), "imagenet_sha256": file_hash(EXISTING_PRETRAINED_PATH),
        "batch_size_source": "author-confirmed release setting; retained script had 16",
        "historical_log_reproduction_verified": False
    }, indent=2), encoding="utf8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=LOG_DIR)

    
    image_files = sorted([
        os.path.join(UNLABELED_DATA_DIR, f)
        for f in os.listdir(UNLABELED_DATA_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])
    print(f"Directory '{UNLABELED_DATA_DIR}': {len(image_files)} unlabeled images.")

    
    rng = random.Random(cfg["seed"])
    rng.shuffle(image_files)
    if len(image_files) < 2:
        raise ValueError("At least two source images are required")
    val_count = max(1, int(len(image_files) * 0.1))
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    import csv
    with open(Path(OUTPUT_DIR) / "ssl_split_manifest.csv", "w", newline="", encoding="utf8") as stream:
        writer_csv = csv.writer(stream); writer_csv.writerow(["filename", "split"])
        for split, paths in [("train", train_files), ("val", val_files)]:
            for item in paths: writer_csv.writerow([Path(item).name, split])
    
    for item in image_files:
        with Image.open(item) as source: source.verify()

    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = QualityRankingPretrainDataset(train_files, transform=train_transform)
    val_dataset = QualityRankingPretrainDataset(val_files, transform=val_transform)

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn
    )

    
    model = QualityAwarePretrainModel(
        pretrained_path=EXISTING_PRETRAINED_PATH,
        device=device,
        drop_path_rate=DROP_PATH_RATE
    ).to(device)

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENCODER_LR},
        {"params": model.deg_type_head.parameters(), "lr": HEADS_LR},
        {"params": model.severity_head.parameters(), "lr": HEADS_LR},
        {"params": model.quality_head.parameters(), "lr": HEADS_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = get_warmup_cosine_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    best_score = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = []

    print("\n--- Starting degradation-aware self-supervised pretraining ---")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Current learning rate: {optimizer.param_groups[0]['lr']:.7f}")

        
        model.train()
        total_loss = 0.0
        total_samples = 0

        deg_correct = 0
        sev_correct = 0
        rank_correct = 0

        pbar = tqdm(train_loader, desc="train")
        for batch in pbar:
            if batch is None:
                continue

            img_a, img_b, type_a, sev_a, type_b, sev_b, rank_label = batch
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            type_a = type_a.to(device)
            sev_a = sev_a.to(device)
            type_b = type_b.to(device)
            sev_b = sev_b.to(device)
            rank_label = rank_label.to(device)

            optimizer.zero_grad()

            (deg_a, sev_logits_a, score_a), (deg_b, sev_logits_b, score_b) = model(img_a, img_b)

            loss_deg = F.cross_entropy(deg_a, type_a) + F.cross_entropy(deg_b, type_b)
            loss_sev = F.cross_entropy(sev_logits_a, sev_a) + F.cross_entropy(sev_logits_b, sev_b)
            rank_logits = score_a - score_b
            loss_rank = F.binary_cross_entropy_with_logits(rank_logits, rank_label)

            loss = 1.0 * loss_deg + 1.0 * loss_sev + 0.5 * loss_rank

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = img_a.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            deg_pred_a = deg_a.argmax(dim=1)
            sev_pred_a = sev_logits_a.argmax(dim=1)
            rank_pred = (torch.sigmoid(rank_logits) > 0.5).long()

            deg_correct += (deg_pred_a == type_a).sum().item()
            sev_correct += (sev_pred_a == sev_a).sum().item()
            rank_correct += (rank_pred == rank_label.long()).sum().item()

        train_loss = total_loss / max(1, total_samples)
        train_deg_acc = deg_correct / max(1, total_samples)
        train_sev_acc = sev_correct / max(1, total_samples)
        train_rank_acc = rank_correct / max(1, total_samples)

        print(f"train Loss: {train_loss:.4f}")
        print(f"train Deg-Type Acc: {train_deg_acc:.4f} | train Severity Acc: {train_sev_acc:.4f} | train Ranking Acc: {train_rank_acc:.4f}")

        
        val_loss, val_deg_acc, val_sev_acc, val_rank_acc, val_score = evaluate_pretrain(model, val_loader, device)

        print(f"val Loss: {val_loss:.4f}")
        print(f"val Deg-Type Acc: {val_deg_acc:.4f} | val Severity Acc: {val_sev_acc:.4f} | val Ranking Acc: {val_rank_acc:.4f}")
        print(f"val Combined Score: {val_score:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_deg_acc": train_deg_acc,
            "train_sev_acc": train_sev_acc,
            "train_rank_acc": train_rank_acc,
            "val_loss": val_loss,
            "val_deg_acc": val_deg_acc,
            "val_sev_acc": val_sev_acc,
            "val_rank_acc": val_rank_acc,
            "val_score": val_score
        })

        
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Acc/train_deg", train_deg_acc, epoch + 1)
        writer.add_scalar("Acc/train_sev", train_sev_acc, epoch + 1)
        writer.add_scalar("Acc/train_rank", train_rank_acc, epoch + 1)
        writer.add_scalar("Acc/val_deg", val_deg_acc, epoch + 1)
        writer.add_scalar("Acc/val_sev", val_sev_acc, epoch + 1)
        writer.add_scalar("Acc/val_rank", val_rank_acc, epoch + 1)
        writer.add_scalar("Score/val_combined", val_score, epoch + 1)
        writer.add_scalar("LR/encoder", optimizer.param_groups[0]["lr"], epoch + 1)

        import pandas as pd
        pd.DataFrame(history).to_csv(
            os.path.join(OUTPUT_DIR, "pretrain_history.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history, "config": cfg
        }, os.path.join(CHECKPOINT_DIR, LAST_CHECKPOINT_FILENAME))

        
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1:03d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history, "config": cfg
            }, ckpt_path)
            print(f"--- Periodic checkpoint saved: {ckpt_path} ---")

        
        if val_score > best_score:
            best_score = val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"*** New best validation combined score: {best_score:.4f} ***")

            best_ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": best_model_wts,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_score": best_score, "config": cfg
            }
            torch.save(best_ckpt, os.path.join(CHECKPOINT_DIR, BEST_CHECKPOINT_FILENAME))
            torch.save(model.encoder.state_dict(), os.path.join(OUTPUT_DIR, BEST_ENCODER_FILENAME))
        else:
            epochs_no_improve += 1
            print(f"  Validation combined score did not improve ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")

        scheduler.step()

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

        print()

    writer.close()

    
    final_encoder_path = os.path.join(OUTPUT_DIR, FINAL_ENCODER_FILENAME)
    model.load_state_dict(best_model_wts)
    torch.save(model.encoder.state_dict(), final_encoder_path)

    print("\n--- Pretraining completed ---")
    print(f"Best encoder saved: {os.path.join(OUTPUT_DIR, BEST_ENCODER_FILENAME)}")
    print(f"Encoder export saved: {final_encoder_path}")
    print(f"Best checkpoint saved: {os.path.join(CHECKPOINT_DIR, BEST_CHECKPOINT_FILENAME)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain.json")
    parser.add_argument("--run", action="store_true", help="Start a new training run")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf8"))
    if not args.run:
        print(json.dumps(cfg, indent=2)); print("No training started. Add --run to start a new run.")
    else:
        pretrain_quality_aware(cfg)
