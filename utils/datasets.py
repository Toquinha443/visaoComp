import os
import cv2
import yaml
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from utils.transforms import full_preprocess

CLASS_TO_IDX = {"none":0, "weak":1, "medium":2, "strong":3}

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def discover_from_folders(root_dir: str, classes: List[str]) -> pd.DataFrame:
    rows = []
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                rows.append({"image_path": os.path.join(cls, fname), "class": cls})
    return pd.DataFrame(rows)

def discover_from_csv(root_dir: str, csv_file: str) -> pd.DataFrame:
    csv_path = os.path.join(root_dir, csv_file)
    df = pd.read_csv(csv_path)
    assert {"image_path","class"}.issubset(df.columns), "CSV precisa ter colunas image_path,class"
    return df

def make_train_val_split(df: pd.DataFrame, val_split: float, seed: int) -> Tuple[pd.DataFrame,pd.DataFrame]:
    return train_test_split(df, test_size=val_split, stratify=df["class"], random_state=seed)

class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir: str, df: pd.DataFrame, cfg: Dict, is_train: bool = True, aug=None):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.is_train = is_train
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Falha ao ler imagem: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pré-processamento
        img = full_preprocess(img, self.cfg)  # float32 [0,1], HxWx3

        # augment (apenas treino)
        if self.is_train and self.aug is not None:
            import albumentations as A
            augmented = self.aug(image=(img*255).astype("uint8"))
            img = augmented["image"].astype("float32")/255.0

        # to tensor CHW
        img = np.transpose(img, (2,0,1))
        x = torch.from_numpy(img).float()

        y = CLASS_TO_IDX[row["class"]]
        y = torch.tensor(y).long()
        return x, y
