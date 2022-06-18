# ZED Social Distance

![Stereolabs ZED](data/helpers/demo.gif)
The demo was run on a laptop with a Nvidia GTX 1060 Mobile with a compute power of 6.1. The demo SVO file is located in data/svo/demo/svo .

## A Project to Compute if People Adhere to Social Distancing

This project computes the real-world distance between people to determine if they are adhering to social distancing practices. The project uses a Stereolabs ZED camera or an SVO file to compute the real-world distance between people. Human detection and tracking are integrated from the projects acknowledged below.

This project was forked from: https://github.com/theAIGuysCode/yolov3_deepsort
and uses technology from:
 - https://github.com/zzh8829/yolov3-tf2
 - https://github.com/nwojke/deep_sort
 - https://arxiv.org/abs/1804.02767


## Requirements
This project requires either a Sterolabs ZED Stereovision camera or a pre-recorded SVO file.
![Stereolabs ZED](data/helpers/zed.jpg)
You will need to install the ZED SDk
 - Getting started guide here: https://www.stereolabs.com/developers/
 - Py-ZED install docs here: https://www.stereolabs.com/docs/app-development/python/install/
 - You will also need to install the SDK's dependencies (inc. Nvidia CUDA)
 - For converting SVO files please see: https://support.stereolabs.com/hc/en-us/articles/360009986754-How-do-I-convert-SVO-files-to-AVI-or-image-depth-sequences-

The "Running the Object Tracker" and "Command Line Args Reference" sections provide examples of using the ZED or an SVO file.


## Getting Started 
These requirements have been curated from the original repo to be able to run YOLOv3, DeepSORT, and Tensorflow using the COCO dataset. (Some tweaking may be required to get them to work with the ZED SDK and CUDA on your system... use of Conda advised.)

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you haven't set it up already)
```bash
# Ubuntu 18.04
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
### Downloading official pretrained weights
For Linux: Let's download official yolov3 weights pretrained on COCO dataset. 

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
```
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and yolov3-tiny [here](https://pjreddie.com/media/files/yolov3-tiny.weights) then save them to the weights folder.

### Using Custom trained weights
<strong> Learn How To Train Custom YOLOV3 Weights Here: https://www.youtube.com/watch?v=zJDUhGL26iU </strong>

Add your custom weights file to weights folder and your custom .names file into data/labels folder.
  
### Saving your yolov3 weights as a TensorFlow model.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .tf model files!

```
# yolov3
python load_weights.py

# yolov3-tiny
python load_weights.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf --tiny

# yolov3-custom (add --tiny flag if your custom weights were trained for tiny model)
python load_weights.py --weights ./weights/<YOUR CUSTOM WEIGHTS FILE> --output ./weights/yolov3-custom.tf --num_classes <# CLASSES>
```

After executing one of the above lines, you should see proper .tf files in your weights folder. You are now ready to run object tracker.

## Running the Object Tracker
You can run the object tracker for whichever model you have created, pretrained, tiny, or custom.
```

#yolov3 on Stereolabs ZED using the median depth and the default 2.0m social distance
python social_distance.py

#yolov3 on Stereolabs ZED with the centerpoint depth and a social distance of 2.0m
python social_distance.py --depth centerpoint --distance 2.0

#yolov3 on the demo SVO File located in data/svo/demo.svo
python social_distance.py --svo ./data/svo/demo.svo

#yolov3-tiny 
python social_distance.py --weights ./weights/yolov3-tiny.tf --tiny

#yolov3-custom (add --tiny flag if your custom weights were trained for tiny model)
python social_distance.py  --weights ./weights/yolov3-custom.tf --num_classes <# CLASSES> --classes ./data/labels/<YOUR CUSTOM .names FILE>
```
Video output is not currently supported.

## Command Line Args Reference
```
load_weights.py:
  --output: path to output
    (default: './weights/yolov3.tf')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './weights/yolov3.weights')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)
    
social_distance.py:
  --classes: path to classes file
    (default: './data/labels/coco.names')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false' - yolov3)
  --depth: centerpoint or median
    (default: 'false' - median)
  --distance: distance in metres/m for social distancing
    (default: '1.5')
    (a float)
  --weights: path to weights file
    (default: './weights/yolov3.tf')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)
  --svo: path to an SVO file
    (default: 'None')
    (without this, the ZED camera is selected as the default mode)
  --yolo_max_boxes: maximum number of detections at one time
    (default: '100')
    (an integer)
  --yolo_iou_threshold: iou threshold for how close two boxes can be before they are detected as one box
    (default: 0.5)
    (a float)
  --yolo_score_threshold: score threshold for confidence level in detection for detection to count
    (default: 0.5)
    (a float)
```

