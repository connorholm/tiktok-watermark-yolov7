from asyncore import read
from fastapi import FastAPI, File, UploadFile
from segmentation import get_yolov7, get_image_from_bytes
from starlette.responses import Response
import io
import os
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import cv2

model = get_yolov7()

app = FastAPI(
    title="TikTok Segmentation API",
    description="Used to Remove the Tiktok Watermark from Videos",
    version="0.1.0",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]    

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message" : "Welcome to the TikTok Watermark Remover API"}

@app.post("/image-to-json")
async def image_to_json(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    return {"results" : detect_res}

@app.post("/image-to-labeled-image")
async def image_to_labeled_image(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render() # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format='jpeg')
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")

@app.post("/remove-watermark")
async def remove_watermark(file: UploadFile = File(...)):
    # directory for the frames to be stored
    frames_dir = "frames-0"
    # checks if the directory exists and changes name if it does
    count = 1
    while os.path.exists(frames_dir):
        frames_dir = frames_dir[:-1] + str(count)
        count += 1
    # creates the directory
    os.mkdir(frames_dir)

    # opens the video file
    video = cv2.VideoCapture(str(file.file))

    # reads the video file
    current_frame = 0
    index = 0
    while(True):
        # read frame
        ret, frame = video.read()

        if ret:
            # run the model on the frame
            results = model(frame)
            results.render() # updates results.imgs with boxes and labels
            for img in results.imgs:
                bytes_io = io.BytesIO()
                img_base64 = Image.fromarray(img)
                img_base64.save(bytes_io, format='jpeg')
            # save frame as JPEG file
            name = './' + frames_dir + str(current_frame) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name, frame)



    return None




