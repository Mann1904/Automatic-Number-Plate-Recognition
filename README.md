#Automatic Number Plate Recognition (ANPR)
Automatic Number Plate Recognition (ANPR) is a system capable of reading vehicle number plates without human intervention through the use of high speed image capture with supporting illumination, detection of characters within the images provided, verification of the character sequences as being those from a vehicle license plate, character recognition to convert image to text; so ending up with a set of metadata that identifies an image containing a vehicle license plate and the associated decoded text of that plate.

## Getting Started
### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2


## Using Custom Trained YOLOv4 Weights

USE MY LICENSE PLATE TRAINED CUSTOM WEIGHTS: https://drive.google.com/file/d/1EUPtbtdF0bjRtNjGv436vDY28EN5DXDH/view?usp=sharing

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.

The only change within the code you need to make in order for your custom model to work is on line 14 of 'core/config.py' file.
Update the code to point at your custom .names file as seen below. (my custom .names file is called custom.names but yours might be named differently)
<p align="center"><img src="data/helpers/custom_config.png" width="640"\></p>

## Custom YOLOv4 Using TensorFlow
The following commands will allow you to run your custom yolov4 model. (video and webcam commands work as well)
```
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 tensorflow model
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car.jpg

# Run yolov4 on video
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/licence_plate.mp4 --output ./detections/results.avi

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi
```

### Crop Detections and Save Them as New Images
I have created a custom function within the file [core/functions.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/functions.py) that can be applied to any detect.py or detect_video.py commands in order to crop the YOLOv4 detections and save them each as their own new image. To crop detections all you need to do is add the `--crop` flag to any command. The resulting cropped images will be saved within the <strong>detections/crop/</strong> folder.
  
 Example of crop flag added to command:
```
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --crop
```
 Here is an example of one of the resulting cropped detections from the above command.
 <p align="center"><img src="data/helpers/crop_example.png" height="250"\></p>
 
<a name="license"/>

## License Plate Recognition Using Tesseract OCR
I have created a custom function to feed Tesseract OCR the bounding box regions of license plates found by my custom YOLOv4 model in order to read and extract the license plate numbers. Thorough preprocessing is done on the license plate in order to correctly extract the license plate number from the image. The function that is in charge of doing the preprocessing and text extraction is called <strong>recognize_plate</strong> and can be found in the file [core/utils.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/utils.py).

<strong>Disclaimer: In order to run tesseract OCR you must first download the binary files and set them up on your local machine. Please do so before proceeding or commands will not run as expected!</strong>

Official Tesseract OCR Github Repo: [tesseract-ocr/tessdoc](https://github.com/tesseract-ocr/tessdoc)

Great Article for How To Install Tesseract on Mac or Linux Machines: [PyImageSearch Article](https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/)

For Windows I recommend: [Windows Install](https://github.com/UB-Mannheim/tesseract/wiki)

Once you have Tesseract properly installed you can move onwards. If you don't have a trained YOLOv4 model to detect license plates feel free to use one that I have trained. It is not perfect but it works well. [Download license plate detector model and learn how to save and run it with TensorFlow here](#custom)

### Running License Plate Recognition on Images (video example below)
The license plate recognition works wonders on images. All you need to do is add the `--plate` flag on top of the command to run the custom YOLOv4 model.

Try it out on this image in the repository!
```
# Run License Plate Recognition
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car2.jpg --plate
```

### Resulting Image Example
The output from the above command should print any license plate numbers found to your command terminal as well as output and save the following image to the `detections` folder.
<p align="center"><img src="data/helpers/lpr_demo.png" width="640"\></p>

You should be able to see the license plate number printed on the screen above the bounding box found by YOLOv4.

### Behind the Scenes
This section will highlight the steps I took in order to implement the License Plate Recognition with YOLOv4 and potential areas to be worked on further.

This demo will be showing the step-by-step workflow on the following original image.
<p align="center"><img src="data/images/car2.jpg" width="640"\></p>

First step of the process is taking the bounding box coordinates from YOLOv4 and simply taking the subimage region within the bounds of the box. Since this image is super small the majority of the time we use cv2.resize() to blow the image up 3x its original size. 
<p align="center"><img src="data/helpers/subimage.png" width="400"\></p>

Then we convert the image to grayscale and apply a small Gaussian blur to smooth it out.
<p align="center"><img src="data/helpers/gray.png" width="400"\></p>

Following this, the image is thresholded to white text with black background and has Otsu's method also applied. This white text on black background helps to find contours of image.
<p align="center"><img src="data/helpers/threshold.png" width="400"\></p>

The image is then dilated using opencv in order to make contours more visible and be picked up in future step.
<p align="center"><img src="data/helpers/dilation.png" width="400"\></p>

Next we use opencv to find all the rectangular shaped contours on the image and sort them left to right.
<p align="center"><img src="data/helpers/contours.png" width="400"\></p>

As you can see this causes many contours to be found other than just the contours of each character within the license plate number. In order to filter out the unwanted regions we apply a couple parameters to be met in order to accept a contour. These parameters are just height and width ratios (i.e. the height of region must be at least 1/6th of the total height of the image). A couple other parameters on area of region etc are also placed. Check out code to see exact details. This filtering leaves us with.
<p align="center"><img src="data/helpers/final.png" width="400"\></p>

The individual characters of the license plate number are now the only regions of interest left. We segment each subimage and apply a bitwise_not mask to flip the image to black text on white background which Tesseract is more accurate with. The final step is applying a small median blur on the image and then it is passed to Tesseract to get the letter or number from it. Example of how letters look like when going to tesseract.
<p align="center"><img src="data/helpers/string.png" width="650"\></p>

Each letter or number is then just appended together into a string and at the end you get the full license plate that is recognized! BOOM!

### Running License Plate Recognition on Video
Running the license plate recognition straight on video at the same time that YOLOv4 object detections causes a few issues. Tesseract OCR is fairly expensive in terms of time complexity and slows down the processing of the video to a snail's pace. It can still be accomplished by adding the `--plate` command line flag to any detect_video.py commands.

However, I believe the best route to go is to run video detections without the plate flag and instead run them with `--crop` flag which crops the objects found on screen and saves them as new images. [See how it works here](#crop) Once the video is done processing at a higher FPS all the license plate images will be cropped and saved within [detections/crop](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/detections/crop/) folder. I have added an easy script within the repository called [license_plate_recognizer.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/license_plate_recognizer.py) that you can run in order to recognize license plates. Plus this allows you to easily customize the script to further enhance any recognitions. I will be working on linking this functionality automatically in future commits to the repository.

Running License Plate Recognition with detect_video.py is done with the following command.
```
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/license_plate.mp4 --output ./detections/recognition.avi --plate
```

The recommended route I think is more efficient is using this command. Customize the rate at which detections are cropped within the code itself.
```
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/license_plate.mp4 --output ./detections/recognition.avi --crop
```

Now play around with [license_plate_recognizer.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/license_plate_recognizer.py) and have some fun!

<a name="ocr"/>


## YOLOv4 Using TensorFlow Lite (.tflite model)
Can also implement YOLOv4 using TensorFlow Lite. TensorFlow Lite is a much smaller model and perfect for mobile or edge devices (raspberry pi, etc).
```bash
# Save tf model for tflite converting
python save_model.py --weights ./data/custom.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 --framework tflite

# yolov4
python convert_tflite.py --weights ./checkpoints/custom-416 --output ./checkpoints/custom-416.tflite

# Run tflite model
python detect.py --weights ./checkpoints/custom-416.tflite --size 416 --model yolov4 --images ./data/images/car.jpg --framework tflite
```
### Result Image (TensorFlow Lite)
You can find the outputted image(s) showing the detections saved within the 'detections' folder.


## YOLOv4 Using TensorRT
Can also implement YOLOv4 using TensorFlow's TensorRT. TensorRT is a high-performance inference optimizer and runtime that can be used to perform inference in lower precision (FP16 and INT8) on GPUs. TensorRT can allow up to 8x higher performance than regular TensorFlow.

# yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom.tf --input_size 416 --model yolov4
python convert_trt.py --weights ./checkpoints/custom.tf --quantize_mode float16 --output ./checkpoints/custom-trt-fp16-416
python detect.py --weights ./checkpoints/custom-trt-fp16-416 --model yolov4 --images ./data/images/car.jpg --framework trt
```

### References  

   Huge shoutout goes to theAIGuysCode for creating the backbone of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/theAIGuysCode/yolov4-custom-functions)
