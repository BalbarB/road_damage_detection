import requests
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
model.eval()
text = "alligator cracking. edge cracking. lateral cracking. longitudinal cracking. ravelling. rutting. striping. pothole."

root_dir = "datasetgd"
sub_folders = ["test", "train", "valid"]
images_dir = "images"
labels_dir = "labelsv1"
os.makedirs(labels_dir, exist_ok=True)

def convert_box_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return [x_center, y_center, width, height]

def train(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 1024))
    img_w, img_h = image.size
    
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.25,
        text_threshold=0.2,
        target_sizes=[image.size[::-1]]
    )

    res = results[0]
    print(res)
    yolo_lines = []

    for class_id, box, score in zip(
        [i for i in range(len(res["text_labels"]))],
        res["boxes"],
        res["scores"]
    ):
        if score < 0.4:
            continue
        yolo_box = convert_box_to_yolo(box.tolist(), img_w, img_h)
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
        yolo_lines = train(img_path)

        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(labels_path, txt_name)
        with open(txt_path, "w") as f:
            for line in yolo_lines:
                f.write(line + "\n")
        
        print(f"Processed {fname}, saved labels to {txt_path}")    
