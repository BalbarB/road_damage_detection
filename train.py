from ultralytics import YOLO

if __name__ == "__main__":

    # Load a model
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="datasetv2/data.yaml", epochs=100, imgsz=640, batch=8, name="road_damage_test_yolo11s", patience=20, workers=4)