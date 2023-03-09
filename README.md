# Hardware Acceleration for AI 2023 Group Project

This is a Python script that performs object segmentation using a pre-trained Fully Convolutional Network (FCN) model with ResNet101 backbone. The script takes an input image and a list of objects to be cropped from the image. The script applies the FCN model to the input image to obtain a segmentation mask and crops out the specified objects from the input image. The script also allows the user to choose a background image to place the cropped objects on. The output image can be saved and displayed using OpenCV. The script uses PyTorch, torchvision, PIL, numpy, argparse, and time modules.

## Setup
Recommended to use Anaconda or Miniconda for running the project.

### Create the environment 

```conda env create --name hwaenv --file=env.yaml```

## Running
```
python gpu_segmentation.py
```

### Command line options

```
-h, --help            show this help message and exit
-c , --crop-image     Path to image of objects to be cropped out
-o  [ ...], --crop-object  [ ...]
                        Object to be cropped from the image. List of objects: ['__background__', 'aeroplane',
                        'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
-bg , --background-image
                        Path to background image to place the cropped objects on. Default is a green background.
-d {cpu,gpu}, --device {cpu,gpu}
                        Device to use (default=cpu)
-s, --save            Save output file
```

### Example
Running the program with GPU with example input images. Saves the output image after finished.
```
python gpu_segmentation.py -c "./images/input/example_input.jpg" -bg "./images/input/scr.png" -d "gpu" -o horse -s
```
