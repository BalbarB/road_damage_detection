from google import genai
from google.genai import types
from PIL import Image
import json
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_api_key(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("GEMINI_API_KEY")

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

classes = ['alligator cracking', 'lateral cracking', 'longitudinal cracking', 'pothole']
client = genai.Client(api_key=load_api_key())
prompt = f"Detect the following road damage types in the image:{classes}. Only include clearly visible damage. Each elements must have: [labels - one of the allowed classes, box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.]."

image = Image.open("test2.jpg")

config = types.GenerateContentConfig(
  response_mime_type="application/json"
  )

response = client.models.generate_content(model="gemini-3-flash-preview",
                                          contents=[image, prompt],
                                          config=config
                                          )

width, height = image.size
bounding_boxes = json.loads(response.text)

converted_bounding_boxes = []
for bounding_box in bounding_boxes:
    class_label = bounding_box["label"]
    abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
    abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
    abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
    abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
    converted_bounding_boxes.append([class_label, abs_x1, abs_y1, abs_x2, abs_y2])

print("Bounding boxes:", converted_bounding_boxes)
draw_boxes(image, converted_bounding_boxes)