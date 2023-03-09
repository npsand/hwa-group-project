import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import FCN_ResNet101_Weights
from PIL import Image
import cv2
import numpy as np
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('-c', '--crop-image', type=str, metavar='', required=True, help='Path to image of objects to be cropped out')
argparser.add_argument('-bg', '--background-image', type=str, metavar='', required=True, help='Path to background image to place the cropped objects on')
argparser.add_argument('-d', '--device', choices=['cpu', 'gpu'], default='cpu', type=str, metavar='', help='Device to use [gpu, cpu] (default=cpu)')
argparser.add_argument('-s', '--save', action='store_true', help='Save output file')
args = argparser.parse_args()


MODEL_WEIGHTS = FCN_ResNet101_Weights.DEFAULT
CLASS_NAMES = MODEL_WEIGHTS.meta['categories']
# https://stackoverflow.com/questions/36459969/how-to-convert-a-list-to-a-dictionary-with-indexes-as-values
CLASS_NAMES_DICT = {k: v for v, k in enumerate(CLASS_NAMES)}


def remove_bg(image, segmented_image):

    # Create mask from segments
    segmented_image = np.array(segmented_image)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]

    # Add green background to the segmented image
    green_bg = np.zeros(image.shape, np.uint8)
    green_bg[:] = (0, 255, 0)

    # Crop image and add a green background
    output_image = cv2.bitwise_and(image, image, dst=green_bg, mask=mask)

    return output_image

# Remove all detected segments except class_name
def filter_class(img_arr, class_name):
    class_int = CLASS_NAMES_DICT[class_name]
    img_arr[img_arr != class_int] = 0
    return img_arr

device = None
if args.device == 'gpu':
    # Load the model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("No GPU available. Using CPU.")
else:
    device = torch.device('cpu')
    print("Using CPU")


model = torch.hub.load('pytorch/vision', 'fcn_resnet101', weights=MODEL_WEIGHTS)
model.to(device)

# Load the image and prepare it for input to the model
input_image = Image.open('./images/input/example_input.jpg')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = preprocess(input_image).unsqueeze(0)
image = image.to(device)

# Run the model on the input image
model.eval()
with torch.no_grad():
    output = model(image)['out'][0]

# Save the output segmentation mask
output_predictions = output.argmax(0)
output_predictions = output_predictions.byte().cpu().numpy()
filtered = filter_class(output_predictions, 'person')

output = Image.fromarray(output_predictions).resize(input_image.size)
output = remove_bg(np.array(input_image, dtype=np.uint8), output)
output = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

if args.save:
    cv2.imwrite('./images/output/example_output.png', output) # Save image in RGB format

cv2.imshow('Output image', output)
if cv2.waitKey(0):
    cv2.destroyAllWindows()