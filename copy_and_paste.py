import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR = "train"
SAVE_DIR = "train_mixed"
NUM_SAMPLES = 200

os.makedirs(SAVE_DIR, exist_ok=True)

sample_dirs = sorted([os.path.join(SRC_DIR, d) for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))])

def load_image_and_masks(sample_path):
    image = cv2.imread(os.path.join(sample_path, "image.tif"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = {}
    for class_id in range(1, 5):
        mask_path = os.path.join(sample_path, f"class{class_id}.tif")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is not None and len(mask.shape) == 2:
                masks[class_id] = mask
    return image, masks

def visualize(image, masks, title="Image"):
    vis = image.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    for idx, (cls, mask) in enumerate(masks.items()):
        color = colors[idx % len(colors)]
        vis[mask > 0] = (np.array(color) * 0.5 + vis[mask > 0] * 0.5).astype(np.uint8)
    plt.figure(figsize=(8,8))
    plt.title(title)
    plt.imshow(vis)
    plt.axis('off')
    plt.show()

def random_copy_and_paste(sample_dirs):
    while True:
        target_path = np.random.choice(sample_dirs)
        target_image, target_masks = load_image_and_masks(target_path)
        target_h, target_w = target_image.shape[:2]

        while True:
            src_path = np.random.choice(sample_dirs)
            if src_path != target_path:
                break
        src_image, src_masks = load_image_and_masks(src_path)

        found = False
        for class_id, src_mask in src_masks.items():
            instance_ids = np.unique(src_mask)
            instance_ids = instance_ids[instance_ids != 0]
            np.random.shuffle(instance_ids)
            for chosen_id in instance_ids:
                binary_mask = (src_mask == chosen_id).astype(np.uint8)
                if (binary_mask[0,:].any() or binary_mask[-1,:].any() or 
                    binary_mask[:,0].any() or binary_mask[:,-1].any()):
                    continue
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    if w > 0 and h > 0:
                        cropped_instance = src_image[y:y+h, x:x+w]
                        cropped_mask = binary_mask[y:y+h, x:x+w]
                        chosen_class = class_id
                        found = True
                        break
            if found:
                break

        if not found:
            continue

        aug_image = target_image.copy()
        aug_masks = {cls: m.copy() for cls, m in target_masks.items()}

        occupied = np.zeros((target_h, target_w), dtype=np.uint8)
        for m in aug_masks.values():
            occupied = np.logical_or(occupied, m > 0)

        max_y = target_h - h
        max_x = target_w - w
        if max_y <= 0 or max_x <= 0:
            continue

        for _ in range(50):
            paste_y = np.random.randint(0, max_y)
            paste_x = np.random.randint(0, max_x)
            if not occupied[paste_y:paste_y+h, paste_x:paste_x+w].any():
                break
        else:
            continue

        if chosen_class not in aug_masks:
            aug_masks[chosen_class] = np.zeros((target_h, target_w), dtype=np.uint8)

        aug_image[paste_y:paste_y+h, paste_x:paste_x+w] = np.where(
            np.expand_dims(cropped_mask, axis=-1) == 1,
            cropped_instance,
            aug_image[paste_y:paste_y+h, paste_x:paste_x+w]
        )

        aug_masks[chosen_class][paste_y:paste_y+h, paste_x:paste_x+w] = np.where(
            cropped_mask == 1,
            np.max(aug_masks[chosen_class]) + 1,
            aug_masks[chosen_class][paste_y:paste_y+h, paste_x:paste_x+w]
        )

        return target_image, target_masks, aug_image, aug_masks

for i in range(NUM_SAMPLES):
    target_image, target_masks, aug_image, aug_masks = random_copy_and_paste(sample_dirs)

    save_path = os.path.join(SAVE_DIR, f"{i:04d}")
    os.makedirs(save_path, exist_ok=True)

    # Save image
    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_path, "image.tif"), aug_image_bgr)

    # Save masks
    for class_id in range(1, 5):
        mask = aug_masks.get(class_id)
        if mask is not None and (mask > 0).any():
            cv2.imwrite(os.path.join(save_path, f"class{class_id}.tif"), mask)
print(f"Copy and paste done, saved to {SAVE_DIR}")
