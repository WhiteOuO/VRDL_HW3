import os
import json
import numpy as np

json_paths = [
    "merged_results/class1/test-results.json",
    "merged_results/class2/test-results.json",
    "merged_results/class3/test-results.json",
    "merged_results/class4/test-results.json"
]

merged_output_path = "merged_results/4model_most/test-results.json"
os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)

all_predictions = []
for path in json_paths:
    with open(path, "r") as f:
        preds = json.load(f)
        all_predictions.extend(preds)

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0

from collections import defaultdict
grouped = defaultdict(list)
for pred in all_predictions:
    grouped[pred["image_id"]].append(pred)

final_predictions = []
iou_threshold = 0.5
for image_id, preds in grouped.items():
    preds.sort(key=lambda x: -x["score"])
    kept = []
    for p in preds:
        should_keep = True
        for kp in kept:
            if compute_iou(p["bbox"], kp["bbox"]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(p)
    final_predictions.extend(kept)

with open(merged_output_path, "w") as f:
    json.dump(final_predictions, f)

