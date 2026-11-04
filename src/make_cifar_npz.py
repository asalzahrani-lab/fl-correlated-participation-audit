"""Build data/cifar10/cifar10_jpeg_mirror.npz from a clone of github.com/YoongiKim/CIFAR-10-images (train/<class>/*.jpg,
test/<class>/*.jpg; the JPEG re-encoding of CIFAR-10).  usage: python3 make_cifar_npz.py /path/to/CIFAR-10-images"""
import sys, os
import numpy as np
from PIL import Image

root = sys.argv[1]; out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cifar10"); os.makedirs(out_dir, exist_ok=True)
classes = sorted(os.listdir(os.path.join(root, "train"))); out = {}
for split in ("train", "test"):
    X, y = [], []
    for ci, c in enumerate(classes):
        d = os.path.join(root, split, c)
        for fn in sorted(os.listdir(d)):
            X.append(np.asarray(Image.open(os.path.join(d, fn)).convert("RGB"), dtype=np.uint8)); y.append(ci)
    out[f"X_{split}"] = np.stack(X); out[f"y_{split}"] = np.array(y, np.int64)
np.savez_compressed(os.path.join(out_dir, "cifar10_jpeg_mirror.npz"), classes=np.array(classes), **out)
print({k: v.shape for k, v in out.items()})
