import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------
# Load model
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map="cuda",
)

CLASSES = [
    'alligator cracking', 
    'lateral cracking', 
    'longitudinal cracking', 
    'pothole']
CLASS_PRIORITY = {
    "pothole": 3,
    "alligator cracking": 2,
    "longitudinal cracking": 1,
    "lateral cracking": 1,
}

# ----------------------------
# Geometry + sanity filter
# ----------------------------
def valid_crack(label, box, img_w, img_h, debug):
    x1, y1, x2, y2 = box
    w = (x2 - x1) * img_w
    h = (y2 - y1) * img_h
    area = w * h

    # too small → hallucination
    if area < 300:
        if debug: print("REJECT area", area)
        return False

    # border hallucinations
    #if x1 < 0.01 or y1 < 0.01 or x2 > 0.99 or y2 > 0.99:
        #if debug: print("REJECT border")
        #return False

    # crack-specific shape constraint
    if label != "pothole":
        aspect = max(w / (h + 1e-6), h / (w + 1e-6))
        if aspect < 2.5:
            if debug: print("REJECT aspect", aspect)
            return False

    return True

# ----------------------------
# IoU
# ----------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0

# ----------------------------
# Cross-class NMS
# ----------------------------
def nms(dets, thr=0.5):
    dets = sorted(
        dets,
        key=lambda d: (
            CLASS_PRIORITY[d["label"]],
            (d["x_max"] - d["x_min"]) * (d["y_max"] - d["y_min"]),
        ),
        reverse=True,
    )
    keep = []

    for d in dets:
        if all(iou(
            (d["x_min"], d["y_min"], d["x_max"], d["y_max"]),
            (k["x_min"], k["y_min"], k["x_max"], k["y_max"]),
        ) < thr for k in keep):
            keep.append(d)

    return keep

# ----------------------------
# YOLO conversion
# ----------------------------
def to_yolo(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h

# ----------------------------
# Detection
# ----------------------------
def detect(image_path):
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    detections = []

    for cls in CLASSES:
        prompt = (
            f"Detect ONLY real, visible {cls} on the road surface. "
            "Ignore shadows, stains, borders, road texture, lane markings, "
            "water, patches, and noise."
        )

        result = model.detect(
            image,
            prompt,
            settings={"max_objects": 10},
        )

        for obj in result.get("objects", []):
            box = (
                obj["x_min"],
                obj["y_min"],
                obj["x_max"],
                obj["y_max"],
            )

            if valid_crack(cls, box, W, H, debug=True):
                detections.append({
                    "label": cls,
                    **obj
                })

    detections = nms(detections)

    # YOLO output
    yolo = []
    for d in detections:
        cls_id = CLASSES.index(d["label"])
        cx, cy, w, h = to_yolo((d["x_min"], d["y_min"], d["x_max"], d["y_max"]))
        yolo.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return detections, yolo

# Visualization

def draw_boxes(image_path, detections):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    W, H = image.size

    for d in detections:
        label = d["label"]

        x1 = d["x_min"] * W
        y1 = d["y_min"] * H
        x2 = d["x_max"] * W
        y2 = d["y_max"] * H

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            label,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.7, pad=2),
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


    # Run
# ======================================================
if __name__ == "__main__":
    image_path = "../../test2.jpg"

    detections, yolo = detect(image_path)

    print("FINAL DETECTIONS:")
    for d in detections:
        print(d)

    draw_boxes(image_path, detections)

    for l in yolo:
        print(l)