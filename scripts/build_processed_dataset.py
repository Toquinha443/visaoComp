import os
import cv2
import yaml
import argparse
import shutil
from glob import glob
from utils.transforms import full_preprocess
from utils.common import ensure_dir

VALID_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")

def process_and_copy(src_path, dst_path, cfg):
    img = cv2.imread(src_path)
    if img is None:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = full_preprocess(img, cfg)
    img = (img*255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst_path, img)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--version", type=str, default="v1", choices=["v1","v2"])
    ap.add_argument("--structure", type=str, default=None, choices=["folders","csv"])
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.structure is not None:
        cfg["dataset"]["structure"] = args.structure

    raw_dir = cfg["data"]["raw_dir"]
    processed_root = cfg["data"]["processed_dir"]
    out_dir = os.path.join(processed_root, args.version)
    if os.path.isdir(out_dir):
        print(f"[Aviso] Limpando saída existente: {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    structure = cfg["dataset"]["structure"]
    classes = cfg["dataset"]["classes"]

    if structure == "folders":
        for cls in classes:
            src_cls_dir = os.path.join(raw_dir, cls)
            dst_cls_dir = os.path.join(out_dir, cls)
            ensure_dir(dst_cls_dir)
            if not os.path.isdir(src_cls_dir):
                print(f"[Aviso] Pasta não encontrada: {src_cls_dir}")
                continue
            for fname in os.listdir(src_cls_dir):
                if not fname.lower().endswith(VALID_EXTS): 
                    continue
                src_path = os.path.join(src_cls_dir, fname)
                dst_path = os.path.join(dst_cls_dir, fname)
                ok = process_and_copy(src_path, dst_path, cfg)
                if not ok:
                    print("Falha em:", src_path)
    else:
        import pandas as pd
        csv_path = os.path.join(raw_dir, cfg["dataset"]["csv_file"])
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel = row["image_path"]; cls = row["class"]
            src_path = os.path.join(raw_dir, rel)
            dst_cls_dir = os.path.join(out_dir, cls)
            ensure_dir(dst_cls_dir)
            fname = os.path.basename(rel)
            dst_path = os.path.join(dst_cls_dir, fname)
            ok = process_and_copy(src_path, dst_path, cfg)
            if not ok:
                print("Falha em:", src_path)

    print(f"Base processada criada em: {out_dir}")

if __name__ == "__main__":
    main()
