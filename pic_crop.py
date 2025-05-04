import os
import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = Path("hw3/train")  
OUTPUT_DIR = Path("train")    
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def split_image_into_quadrants(img):
    h, w = img.shape[:2]
    mh, mw = h // 2, w // 2
    return [
        img[0:mh, 0:mw],      # top left
        img[0:mh, mw:w],      # top right
        img[mh:h, 0:mw],      # bottem left
        img[mh:h, mw:w],      # bottom right
    ]

sample_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])

for idx, sample_dir in enumerate(sample_dirs, 1):
    image_path = sample_dir / "image.tif"
    if not image_path.exists():
        continue

    image = cv2.imread(str(image_path))
    image_quads = split_image_into_quadrants(image)

    for part_idx, img_part in enumerate(image_quads, 1):
        part_dir = OUTPUT_DIR / f"{idx}_{part_idx}"
        part_dir.mkdir(parents=True, exist_ok=True)

        image_out_path = part_dir / "image.tif"
        cv2.imwrite(str(image_out_path), img_part)

    for class_id in range(1, 5):
        mask_path = sample_dir / f"class{class_id}.tif"
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) 
        mask_quads = split_image_into_quadrants(mask)

        for part_idx, mask_part in enumerate(mask_quads, 1):
            part_dir = OUTPUT_DIR / f"{idx}_{part_idx}"
            part_dir.mkdir(parents=True, exist_ok=True)

            mask_out_path = part_dir / f"class{class_id}.tif"
            cv2.imwrite(str(mask_out_path), mask_part)

print("image divide done")
