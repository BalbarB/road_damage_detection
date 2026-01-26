from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

if __name__ == "__main__":
    benchmark(model="models/test_model_sv2.pt", data="datasetv2/data.yaml", imgsz=640, half=False, device=0)