from ultralytics import YOLO

if __name__ == "__main__":
    # Load your best trained model
    model = YOLO("models/test_model_s.pt")

    metrics = model.val()
    print(metrics.box.map) 
    
    # Run inference on an image
    # results = model(["test1.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
    # for result in results:
    #    boxes = result.boxes  # Boxes object for bounding box outputs
    #    masks = result.masks  # Masks object for segmentation masks outputs
    #    keypoints = result.keypoints  # Keypoints object for pose outputs
    #    probs = result.probs  # Probs object for classification outputs
    #    obb = result.obb  # Oriented boxes object for OBB outputs
    #    result.show()  # display to screen
    #    result.save(filename="result_s_v1.jpg")  # save to disk