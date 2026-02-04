from transformers import AutoModelForCausalLM
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map="cuda",
)
classes = ['alligator cracking', 'lateral cracking', 'longitudinal cracking', 'pothole']
# Load your image
image = Image.open("../../test1.jpg")

# Optional sampling settings
settings = {"max_objects": 50}
detections = []


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


# Run Moondream
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

for d in detections:
    print(d)

draw_boxes(image, detections)


