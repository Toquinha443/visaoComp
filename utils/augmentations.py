import albumentations as A

def build_train_augment(cfg):
    aug_list = []
    if cfg["augment"]["horizontal_flip"]:
        aug_list.append(A.HorizontalFlip(p=0.5))
    if cfg["augment"]["rotate_limit"]:
        aug_list.append(A.Rotate(limit=cfg["augment"]["rotate_limit"], p=0.5))
    if cfg["augment"]["brightness_contrast"]:
        b, c = cfg["augment"]["brightness_contrast"]
        aug_list.append(A.RandomBrightnessContrast(brightness_limit=b, contrast_limit=c, p=0.5))
    if cfg["augment"]["blur_prob"] > 0:
        aug_list.append(A.Blur(blur_limit=3, p=cfg["augment"]["blur_prob"]))
    if cfg["augment"]["noise_prob"] > 0:
        aug_list.append(A.GaussNoise(var_limit=(5.0, 20.0), p=cfg["augment"]["noise_prob"]))
    return A.Compose(aug_list)
