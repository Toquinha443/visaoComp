import os
import yaml
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from sklearn.metrics import classification_report, confusion_matrix

from utils.common import set_seed
from utils.datasets import (
    load_config, discover_from_folders, discover_from_csv, make_train_val_split,
    ImageClassificationDataset
)
from utils.augmentations import build_train_augment

def build_model(num_classes: int, pretrained: bool = True):
    model = mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(preds.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return running_loss/total, correct/total, ys, ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--val_split", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None: cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None: cfg["train"]["batch_size"] = args.batch_size
    if args.val_split is not None: cfg["train"]["val_split"] = args.val_split
    if args.seed is not None: cfg["train"]["seed"] = args.seed

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data dir
    data_dir = args.data_dir or os.path.join(cfg["data"]["processed_dir"], cfg["data"]["version"])

    # Descobrir imagens (processed v1/v2 em pastas por classe)
    classes = cfg["dataset"]["classes"]
    if all(os.path.isdir(os.path.join(data_dir, c)) for c in classes):
        df = discover_from_folders(data_dir, classes)
    else:
        csv_path = os.path.join(data_dir, "labels.csv")
        if os.path.isfile(csv_path):
            df = discover_from_csv(data_dir, "labels.csv")
        else:
            raise RuntimeError("Estrutura esperada não encontrada (pastas por classe em processed ou CSV).")

    # Split
    train_df, val_df = make_train_val_split(df, cfg["train"]["val_split"], cfg["train"]["seed"])

    # Augment
    aug = build_train_augment(cfg)

    # Datasets
    train_ds = ImageClassificationDataset(data_dir, train_df, cfg, is_train=True, aug=aug)
    val_ds   = ImageClassificationDataset(data_dir, val_df,   cfg, is_train=False, aug=None)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True)

    # Model
    model = build_model(num_classes=len(classes), pretrained=cfg["train"]["pretrained"]).to(device)

    # Optim + Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    if cfg["train"]["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    elif cfg["train"]["lr_scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # Training loop
    best_val_acc = 0.0
    patience = cfg["train"]["early_stopping_patience"]
    patience_ctr = 0
    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt = os.path.join(run_dir, "best_model.pt")

    for epoch in range(cfg["train"]["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1:03d}/{cfg['train']['epochs']} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        # Early stopping + checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "classes": classes}, best_ckpt)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping ativado.")
                break

    print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
    print(f"Checkpoint salvo em: {best_ckpt}")

    # Relatório final
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, _, y_true, y_pred = validate(model, val_loader, criterion, device)
    print("\nClassification Report (val):")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print("Confusion Matrix (val):")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
