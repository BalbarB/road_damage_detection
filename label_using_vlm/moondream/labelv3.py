import os
from transformers import AutoModelForCausalLM
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load model directly
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True, 
    device_map="cuda",
)

classes = ['alligator cracking', 'lateral cracking', 'longitudinal cracking', 'pothole']
PHRASE_TO_CLASS = {
    "alligator cracking": 0,
    "lateral cracking": 1,
    "longitudinal cracking": 2,
    "pothole": 3
}

# Load your image
root_dir = "../../dataset"
sub_folders = ["valid"]
images_dir = "images"
labels_dir = "labels_MD"
os.makedirs(labels_dir, exist_ok=True)

# Optional sampling settings
settings = {"max_objects": 50}

def draw_boxes(image, detections):
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

def annotate(image_path):
    image = Image.open(image_path).convert("RGB")
    detections = []

    for cls in classes:
        result = model.detect(image, cls, settings=settings)
        print(f"Class = {cls}")
        print(result)
        print("-" * 40)
        for obj in result.get("objects", []):
            box = (
                obj["x_min"],
                obj["y_min"],
                obj["x_max"],
                obj["y_max"],
            )
            detections.append({
                "label": cls,
                **obj
            })

    #for d in detections:
    #    print(d)

    #draw_boxes(image, detections)

    yolo_lines = []
    for d in detections:
        class_id = PHRASE_TO_CLASS[d["label"]]
        x_center = (d["x_min"] + d["x_max"]) / 2
        y_center = (d["y_min"] + d["y_max"]) / 2
        width = d["x_max"] - d["x_min"]
        height = d["y_max"] - d["y_min"]
        yolo_box = [x_center, y_center, width, height]

        line = f"{class_id} " + " ".join(f"{x:.6f}" for x in yolo_box)
        yolo_lines.append(line)
    return yolo_lines

for split in sub_folders:
    images_path = os.path.join(root_dir, split, images_dir)
    labels_path = os.path.join(root_dir, split, labels_dir)
    os.makedirs(labels_path, exist_ok=True)
    
    for fname in os.listdir(images_path):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(images_path, fname)
        yolo_lines = annotate(img_path)

        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(labels_path, txt_name)
        with open(txt_path, "w") as f:
            for line in yolo_lines:
                f.write(line + "\n")
        
        print(f"Processed {fname}, saved labels to {txt_path}")    


