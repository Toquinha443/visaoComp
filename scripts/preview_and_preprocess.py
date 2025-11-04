import os
import cv2
import yaml
import argparse
from glob import glob
from utils.transforms import full_preprocess
from utils.common import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--n_samples", type=int, default=20)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["data"]["raw_dir"]
    interim_dir = cfg["data"]["interim_dir"]
    ensure_dir(interim_dir)

    # coleta poucas imagens para visual
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    imgs = []
    for ext in exts:
        imgs.extend(glob(os.path.join(raw_dir, "**", ext), recursive=True))
    imgs = imgs[:args.n_samples]
    print(f"Gerando prévias para {len(imgs)} imagens...")

    for i, path in enumerate(imgs):
        img = cv2.imread(path)
        if img is None:
            print("falha:", path); 
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        proc = full_preprocess(img, cfg)

        # salvar lado a lado (original vs processada)
        vis = (img*255).astype("uint8") if img.max()<=1.0 else img.copy()
        out = (proc*255).astype("uint8") if proc.max()<=1.0 else proc.copy()
        side = cv2.hconcat([vis, out])
        out_path = os.path.join(interim_dir, f"preview_{i:04d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    print(f"Pré-processamento de amostra salvo em {interim_dir}")

if __name__ == "__main__":
    main()
