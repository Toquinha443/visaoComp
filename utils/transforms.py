import cv2
import numpy as np

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_denoise(img, method="bilateral", **kwargs):
    if method == "bilateral":
        d = kwargs.get("d", 7); sigmaColor = kwargs.get("sigmaColor", 50); sigmaSpace = kwargs.get("sigmaSpace", 50)
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    elif method == "median":
        k = kwargs.get("ksize", 5); return cv2.medianBlur(img, k)
    elif method == "gaussian":
        k = kwargs.get("ksize", 5); sigmaX = kwargs.get("sigmaX", 0)
        return cv2.GaussianBlur(img, (k, k), sigmaX)
    return img

def resize_and_normalize(img, size=(512,512), to_grayscale=False, normalize=True):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if to_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # manter 3 canais
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img

def full_preprocess(img, cfg):
    # Resize
    img = resize_and_normalize(
        img,
        size=tuple(cfg["preprocess"]["resize"]),
        to_grayscale=cfg["preprocess"]["to_grayscale"],
        normalize=cfg["preprocess"]["normalize"]
    )
    # CLAHE
    if cfg["preprocess"]["clahe"]["enabled"]:
        img_uint8 = (img*255).astype("uint8")
        img_uint8 = apply_clahe(
            img_uint8,
            clip_limit=cfg["preprocess"]["clahe"]["clip_limit"],
            tile_grid_size=tuple(cfg["preprocess"]["clahe"]["tile_grid_size"]),
        )
        img = img_uint8.astype("float32")/255.0
    # Denoise
    if cfg["preprocess"]["denoise"]["enabled"]:
        method = cfg["preprocess"]["denoise"]["method"]
        params = cfg["preprocess"]["denoise"].get(method, {})
        img_uint8 = (img*255).astype("uint8")
        img_uint8 = apply_denoise(img_uint8, method=method, **params)
        img = img_uint8.astype("float32")/255.0
    return img
