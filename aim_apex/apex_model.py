import torch
from models.common import DetectMultiBackend
from models.experimental import attempt_load

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def model(weights, imgsz=640, half=False):
    model = DetectMultiBackend(weights, device=device)
    model.half() if half else model.float()
    model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)
    return model