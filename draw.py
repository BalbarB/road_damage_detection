from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

image_path = "test1.jpg"
image = Image.open(image_path).convert("RGB")

boxes = [
    ['pothole', 171, 201, 195, 212],
    ['alligator cracking', 91, 238, 164, 264],
    ['pothole', 265, 248, 320, 267],
    ['pothole', 153, 283, 211, 309]
]

def draw_boxes(image, boxes):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for label, x1, y1, x2, y2 in boxes:

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"{label}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.6)
        )

    plt.axis("off")
    plt.show()


draw_boxes(image, boxes)