import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import FCN_ResNet101_Weights
from PIL import Image
import cv2
import numpy as np

MODEL_WEIGHTS = FCN_ResNet101_Weights.DEFAULT
CLASS_NAMES = MODEL_WEIGHTS.meta['categories']
# https://stackoverflow.com/questions/36459969/how-to-convert-a-list-to-a-dictionary-with-indexes-as-values
CLASS_NAMES_DICT = {k: v for v, k in enumerate(CLASS_NAMES)}

def image_overlay(image, segmented_image):
    segmented_image = np.array(segmented_image)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # add green background to the segmented image
    #segmented_image[mask == [0,0,0]] = [0, 255, 0]
    imagee = np.zeros(image.shape, np.uint8)
    imagee[:] = (0, 255, 0)

    output_image = cv2.bitwise_and(image, image, dst=imagee, mask=mask)

    return output_image


def filter_class(img_arr, class_name):
    class_int = CLASS_NAMES_DICT[class_name]
    img_arr[img_arr != class_int] = 0
    return img_arr

# Load the model on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU")
else:
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
r = Image.fromarray(output_predictions).resize(input_image.size)
r = image_overlay(np.array(input_image, dtype=np.uint8), r)
#r.putpalette(colors)
cv2.imwrite('./images/output/example_output.png', r[:,:,::-1]) # Save image in RGB format
#cv2.imshow('output', r[:,:,::-1])