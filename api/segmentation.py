import torch
from PIL import Image
import io
import pandas as pd

def get_yolov7():
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load("./yolov7", "custom", "./yolov7/yolov7-tiktok-v1.pt", force_reload=True, trust_repo=True, source="local")

    if torch.cuda.is_available():
        model.cuda()
    model.conf = 0.5
    model.eval()
    return model

def get_image_from_bytes(binary_image, max_size = 1024):
    input_image = Image.open(io.BytesIO(binary_image))
    width, height = input_image.size
    resize_factor = min(max_size/width, max_size/height)
    resized_image = input_image.resize((int(width*resize_factor), int(height*resize_factor)))
    return resized_image

