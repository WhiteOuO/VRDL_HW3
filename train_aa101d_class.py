import os
import cv2
import gc
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
from torch.optim import AdamW
import timm
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from model import CustomBackbone
import tifffile
from PIL import Image
import matplotlib.pyplot as plt

class InstanceSegDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.uuid_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.image_paths = []
        self.image_ids = []

        for idx, uuid in enumerate(self.uuid_dirs):
            img_path = os.path.join(root_dir, uuid, "image.tif")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.image_ids.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = self.image_ids[idx]
        uuid = os.path.basename(os.path.dirname(img_path))

        # Read and convert image
        image = tifffile.imread(img_path)
        image = Image.fromarray(image).convert("RGB")
        image_tensor = F.to_tensor(image)

        masks = []
        boxes = []
        labels = []

        mask_path = os.path.join(self.root_dir, uuid, f"class1.tif")
        if os.path.exists(mask_path):
            mask = tifffile.imread(mask_path)
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]  # skip background
            for inst_id in instance_ids:
                instance_mask = (mask == inst_id).astype(np.uint8)
                if instance_mask.sum() == 0:
                    continue
                y_indices, x_indices = np.where(instance_mask)
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
                if y_max > y_min and x_max > x_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    masks.append(instance_mask)
                    labels.append(1)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.uint8),
                "image_id": torch.tensor([img_id])
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id])
            }

        return image_tensor, target


def get_resnetaa101d_maskrcnn(num_classes=2):
    model = timm.create_model('resnetaa101d.sw_in12k_ft_in1k', pretrained=True, features_only=False)
    backbone = CustomBackbone(model)
    backbone_with_fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
        backbone,
        return_layers={"layer1": "1", "layer2": "2", "layer3": "3", "layer4": "4"},
        in_channels_list=backbone.out_channels,
        out_channels=256
    )
    anchor = AnchorGenerator(
        sizes=((8, 16), (16, 32), (32, 64), (64, 128)), 
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )
    box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['1', '2', '3', '4'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['1', '2', '3', '4'], output_size=14, sampling_ratio=2)
    
    model = MaskRCNN(backbone_with_fpn, num_classes=2, rpn_anchor=anchor,
                     box_roi_pool=box_roi_pooler, mask_roi_pool=mask_roi_pooler, box_detections_per_image=1000)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnetaa101d_maskrcnn(num_classes=2).to(device)
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=5e-4)
scaler = torch.cuda.amp.GradScaler()

num_epochs = 100
model.train()
best_loss = float('inf')
early_stop_counter = 0
max_early_stop = 7

accumulate_steps = 8

for epoch in range(num_epochs):
    dataset = InstanceSegDataset(root_dir="train_by_class/class1")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(tqdm(dataloader, desc=f"[Epoch {epoch+1}]")):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type='cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) / accumulate_steps

        scaler.scale(losses).backward()

        if step % accumulate_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses.item() * accumulate_steps

        del step, images, targets, loss_dict, losses
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        print("New best model, resetting early_stop_counter")
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f"best_model_class1_epoch{epoch+1:03d}_loss{avg_loss:.4f}.pth")
    else:
        early_stop_counter += 1
        print(f"No improvement, early_stop_counter += 1 → {early_stop_counter}")
        if early_stop_counter % 3 == 0:
            if optimizer.param_groups[0]['lr'] > 1e-6:
                optimizer.param_groups[0]['lr'] *= 0.6
                print(f"Reduce lr to {optimizer.param_groups[0]['lr']:.6f}")
            else:
                optimizer.param_groups[0]['lr'] = 1e-6

    if early_stop_counter >= max_early_stop:
        print("Early stopping triggered, exiting training loop.")
        break

    del dataset, dataloader
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(20)
