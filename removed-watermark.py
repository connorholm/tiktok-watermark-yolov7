import torch
from PIL import Image
import io
import os
from segmentation import get_yolov7
import cv2
import pandas as pd
import json
import numpy as np
import math
import ffmpeg
import time

video_file = "./testing/test.mp4"
output_file = "./testing/output.mp4"

# starts program timer
start = time.time()


model = get_yolov7()

# directory for the frames to be stored
frames_dir = "frames-0"
# checks if the directory exists and changes name if it does
count = 1
while os.path.exists(frames_dir):
    frames_dir = frames_dir[:-(math.floor(math.log10(count))+1)] + str(count)
    count += 1
# creates the directory
os.mkdir(frames_dir)

# opens the video file
video = cv2.VideoCapture(video_file)

# reads the video file
current_frame = 0
index = 0
while(True):
    ret, frame = video.read()


    if ret:
        # run the model and get the bounding boxes
        results = model(frame)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")
        detect_res = json.loads(detect_res)
        
        # create a mask that will have the bounding boxes
        # the mask will use inpainting to remove the watermark
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in detect_res:
            x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
            # draw the filled rectangle on the mask 
            mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            mask = mask.astype(np.uint8)

        # inpaint the frame
        frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        # save frame as JPEG file
        name = './' + frames_dir + '/' + str(current_frame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        current_frame += 1
    else:
        break
video.release()
cv2.destroyAllWindows()

print("Adding frames to video...")

# create the video from the frames in the directory
img_array = []
for filename in os.listdir(frames_dir):
    img = cv2.imread(frames_dir + '/' + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# remove the frames directory even if it is not empty
for filename in os.listdir(frames_dir):
    os.remove(frames_dir + '/' + filename)
os.rmdir(frames_dir)

# add the audio to the video
try:
    # os.system("ffmpeg -i " + video_file + " -i " + output_file + " -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 " + output_file) 
    video = ffmpeg.input(output_file)
    audio = ffmpeg.input(video_file)
    ffmpeg.output(video, audio, output_file, vcodec='copy', acodec='aac', strict='experimental').run()
except:
    print("Error adding audio to the video")

# ends program timer
end = time.time()
print("Time elapsed: " + str(round(end - start, 2)) + " seconds")

