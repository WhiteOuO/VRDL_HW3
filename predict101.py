import os
import json
import torch
import torchvision.transforms.functional as F
from pycocotools import mask as mask_utils
import numpy as np
import cv2
from tqdm import tqdm

import timm
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CustomBackbone(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.stem = nn.Sequential(*list(base_model.children())[:4])
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.out_channels = [256, 512, 1024, 2048] 

        def forward(self, x):
            x = self.stem(x)
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            return {
                "layer1": c2,
                "layer2": c3,
                "layer3": c4,
                "layer4": c5,
            }
def get_resnetaa101d_maskrcnn(num_classes=5):
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
    
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes, rpn_anchor=anchor,
                     box_roi_pool=box_roi_pooler, mask_roi_pool=mask_roi_pooler, box_detections_per_image=1000)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

MODELS_DIR = "model_class4"    # file of all models
TEST_DIR = "test" 
TEST_META = "test_image_name_to_ids.json" 
OUTPUT_DIR = "output_class4"   # output will be "{OUTPUT_DIR}/model-name/test-results.json"

with open(TEST_META, 'r') as f:
    test_info = json.load(f)

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
print(f"Found {len(model_files)} model(s) to predict.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_file in model_files:
    model_path = os.path.join(MODELS_DIR, model_file)
    model_name = os.path.splitext(model_file)[0]
    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\nLoading model: {model_file}")
    model = get_resnetaa101d_maskrcnn(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    results = []
    for img_info in tqdm(test_info, desc=f"Predicting with {model_name}"):
        img_path = os.path.join(TEST_DIR, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = img_info["height"], img_info["width"]
        tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            outputs = model([tensor])[0]

        for i in range(len(outputs["scores"])):
            score = outputs["scores"][i].item()
            if score < 0.5:
                continue
            label = outputs["labels"][i].item()
            box = outputs["boxes"][i].tolist()
            mask = outputs["masks"][i, 0].cpu().numpy() > 0.5
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            results.append({
                "image_id": img_info["id"],
                "category_id": label,
                "bbox": [
                    box[0], box[1],
                    box[2] - box[0],
                    box[3] - box[1]
                ],
                "score": score,
                "segmentation": {
                    "size": [height, width],
                    "counts": rle["counts"]
                }
            })

    output_json_path = os.path.join(model_output_dir, "test-results.json")
    with open(output_json_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved predictions to {output_json_path}")

print("\nAll models finished prediction.")
