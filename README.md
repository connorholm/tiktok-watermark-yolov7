# Tiktok Watermark Remover
<img src="./tiktok-logo.png"  width="30%" height="30%">
</br>
Using machine build a powerful tool that will be able to remove watermarks from downloaded Tiktok videos.
</br>
**Example:**
* Input Video with Watermark ![input video](./testing/test.mp4)
* Output Video without Watermark ![output video](./testing/output.mp4)



## Steps for Removing Watermarks
### Building the Model
1. Creating a custom dataset of tiktok videos with watermarks
2. Hand label the dataset with bounding boxes around the watermarks
3. Fine tune the [YoloV7](https://github.com/WongKinYiu/yolov7) object detection model to find watermarks

### Removing the Watermarks
1. Spliting a video into frames
2. Running the fine-tuned model on each frame to find the bounding box around a watermark (if it exists)
3. Creating a mask of the watermark
4. Using [OpenCV's inpaint](https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html) method to remove the watermark from the frame
5. Stitching the frames back together to create a watermark free video


## Installation
Make sure that you have `conda` installed
* run commands from inside the projects root directory

### Creating the Virtual Enviornment
* Run: `conda create -n watermark-remover` to create the new virtual enviornment
* Install pip with `conda install pip`
* Add all the required packages with `pip install -r requirements.txt`

## Testing Functionallity
The time it takes may vary extremely depending on the size of the video and the type of gpu you have. For reference, I have a RTX 3070ti and it finishes in about 10 seconds for a 12 second video.
* Run: `python remove_watermark.py` to test the functionallity of the watermark remover.

## API
I am currently looking to build out this as a REST API. 
* If you would like to test it out, run the following command inside of the `./api` directory: `unvicorn main:app --reload --host 0.0.0.0 --port 8000`
> Note: This is still in progress so some endpoints may not work yet


## Areas of Improvement
* The inpainting model doesn't do a great job of removing the watermark and leaving the images undetectable. I am currently looking into other methods of removing the watermark that will leave the image looking more natural.
* Using a more advanced object detection labeling system that would be able to separate the watermark logo and username apart. This should improve the area around what is being inpainted on.