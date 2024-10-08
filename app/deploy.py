import io
import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from google.cloud import storage

import training.neural_network as neural_network

app = Flask(__name__)
client = storage.Client(project="neural network")
bucket = client.get_bucket("continental_neural_network")
blob = bucket.blob("outputs/final_model.pth")
stored_state_dict = blob.download_as_string()
buffer = io.BytesIO(stored_state_dict)
model = neural_network.NeuralNetwork()
model.load_state_dict(torch.load(buffer,map_location=torch.device('cpu')))
#model = model(pretrained=True)               # Trained on 1000 classes from ImageNet
model.eval()                                              # Turns off autograd


img_class_map = None
mapping_file_path = os.path.abspath("./outputs/labels.json")                 # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

print(img_class_map)

# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [#transforms.Resize(28),           # We use multiple TorchVision transforms to ready the image
        #transforms.CenterCrop(224),
        transforms.ToTensor(),]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg


# Get a prediction
def get_prediction(input_tensor):
    with torch.no_grad():
        outputs = model.forward(input_tensor)                # Get likelihoods for all ImageNet classes
        _, y_hat = outputs.max(1)                             # Extract the most likely class
        prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
    return prediction

# Make the prediction human-readable
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx]

    return prediction_idx, class_name


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()