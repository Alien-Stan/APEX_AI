import numpy as np
from utils.augmentations import letterbox


def img_porcess(img, model, imgsz):
    stride = model.stride
    img = letterbox(img, imgsz, stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return img
