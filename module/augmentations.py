import cv2
import numpy as np
import random
from typing import List


def gamma_correct(img_rgb: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    img = img_rgb.astype(np.float32) / 255.0
    img = img ** (1.0 / gamma)
    img = (img * 255).clip(0, 255).astype("uint8")
    return img


def gaussian_blur(img_rgb: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(img_rgb, (k, k), 0)


def hflip(img_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.fliplr(img_rgb))


def apply_clahe_rgb(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return rgb


def tta_augment(img_rgb: np.ndarray) -> List[np.ndarray]:
    base = img_rgb.copy()
    return [
        base,
        gamma_correct(base, gamma=1.5),
        gamma_correct(base, gamma=1.8),
        apply_clahe_rgb(base),
        gaussian_blur(base, k=3),
        hflip(base),
    ]


def dark_augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    aug = img.copy()
    gamma = random.uniform(0.15, 0.4)
    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)])).astype("uint8")
    aug = cv2.LUT(aug, table)
    brightness_factor = random.uniform(0.2, 0.45)
    aug = np.clip(aug * brightness_factor, 0, 255).astype(np.uint8)
    noise_std = random.randint(3, 8)
    noise = np.random.normal(0, noise_std, (h, w, 3))
    aug = np.clip(aug + noise, 0, 255).astype(np.uint8)
    aug = cv2.GaussianBlur(aug, (3, 3), 0)
    shift_b = random.randint(-10, -3)
    aug[..., 0] = np.clip(aug[..., 0] + shift_b, 0, 255)
    shift_r = random.randint(3, 10)
    aug[..., 2] = np.clip(aug[..., 2] + shift_r, 0, 255)
    if random.random() < 0.5:
        aug = cv2.flip(aug, 1)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aug


def apply_clahe_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def light_augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    if random.random() < 0.5:
        kernel_x = cv2.getGaussianKernel(w, w/2)
        kernel_y = cv2.getGaussianKernel(h, h/2)
        mask = kernel_y @ kernel_x.T
        mask = mask / mask.max()
        mask = 0.9 + 0.1 * mask
        vignette = (out * mask[..., None]).astype(np.uint8)
        out = vignette
    if random.random() < 0.5:
        cx = random.randint(w//4, 3*w//4)
        cy = random.randint(h//4, 3*h//4)
        radius = random.randint(int(0.2*min(h,w)), int(0.4*min(h,w)))
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = 1 + 0.15 * np.clip(1 - dist/radius, 0, 1)
        light = np.clip(out * mask[..., None], 0, 255).astype(np.uint8)
        out = light
    if random.random() < 0.5:
        x1 = random.randint(0, w//2)
        y1 = random.randint(0, h//2)
        x2 = random.randint(w//2, w)
        y2 = random.randint(h//2, h)
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(shadow_mask, (x1, y1), (x2, y2), 1, -1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (99, 99), 30)
        shadow = 1 - 0.15 * shadow_mask
        out = np.clip(out * shadow[..., None], 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out
