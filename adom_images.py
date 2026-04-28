import math
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import adom_library as adom

transform = T.Compose([
    T.ToTensor()
])

random.seed(42)


def preprocess_pil_image(img):
    return transform(img)


def classify_pil_image(model, img, classes):
    img_tensor = transform(img)
    adom.predict_top5_classes(model, img_tensor, classes)


def apply_noise(img, std=25):
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, std, arr.shape)
    noisy = arr + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_occlusion(img, size_ratio=0.3, position="center"):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    w, h = img.size
    occ_w = int(w * size_ratio)
    occ_h = int(h * size_ratio)

    if position == "center":
        x1 = (w - occ_w) // 2
        y1 = (h - occ_h) // 2
    elif position == "random":
        x1 = random.randint(0, w - occ_w)
        y1 = random.randint(0, h - occ_h)
    else:
        raise ValueError("position should be 'center' or 'random'")

    x2 = x1 + occ_w
    y2 = y1 + occ_h

    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    return img


def apply_blur(img, radius=2):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness(img, factor=0.5):
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_contrast(img, factor=0.5):
    return ImageEnhance.Contrast(img).enhance(factor)


def apply_rotation(img, angle=30):
    return img.rotate(angle)


def save_image_variants(variants, folder="modified_images"):
    os.makedirs(folder, exist_ok=True)

    for name, img in variants.items():
        img.save(os.path.join(folder, f"{name}.png"))


def load_image_variants(folder="modified_images"):
    variants = {}
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(folder, filename))
            variants[name] = img
    return variants


def load_external_images(folder="external_images"):
    images = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            name = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images[name] = img
    return images


def show_structured_images(variants, cols=4, figsize=(12, 10)):
    n = len(variants)
    rows = math.ceil(n / cols)

    plt.figure(figsize=figsize)

    for i, (name, img) in enumerate(variants.items()):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
