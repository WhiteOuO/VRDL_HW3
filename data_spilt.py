import os
import shutil
import numpy as np

SRC_DIR = "train_mixed" 
SAVE_DIR = "train_mixed_spilt" 
NUM_SPLITS = 10         

sample_dirs = sorted([d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))])

np.random.seed(42) 
np.random.shuffle(sample_dirs)

splits = [[] for _ in range(NUM_SPLITS)]
for idx, folder in enumerate(sample_dirs):
    splits[idx % NUM_SPLITS].append(folder)

for i, split in enumerate(splits):
    split_dir = os.path.join(SAVE_DIR, f"train_part_{i}")
    os.makedirs(split_dir, exist_ok=True)
    for folder in split:
        src_path = os.path.join(SRC_DIR, folder)
        dst_path = os.path.join(split_dir, folder)
        shutil.copytree(src_path, dst_path)

