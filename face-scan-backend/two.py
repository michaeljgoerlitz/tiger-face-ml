import torch
import pickle
import torchvision.models as models
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from mtcnn import MTCNN
import os

import warnings
warnings.filterwarnings("ignore")

# Load pre-trained model like ResNet50
# model = models.resnet50(pretrained=True)
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Get model ready for feature extraction, removing last layer
model = torch.nn.Sequential(*(list(model.children())[:-1]))

feature_list = pickle.load(open('embedding.pkl', 'rb'))

def get_file_dict(folder_path):
    file_to_num = {}
    curr_file = 0

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        file_to_num[curr_file] = file_path
        curr_file += 1

    return file_to_num


def extract_features(img_path, model, detector):
    # Read the image
    img = Image.open(img_path)

    # img_array = np.array(img)
    # print("image array: ", img_array)

    # # Detect faces
    # results = detector.detect_faces(img_array)

    # print('Results: ', results)
    # # Assuming the first detected face is used
    # x, y, width, height = results[0]['box']
    # face = img.crop((x, y, x + width, y + height))

    face = img

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformation to the face
    face_tensor = transform(face)

    # Add an extra batch dimension since PyTorch models expect batched input
    face_tensor = face_tensor.unsqueeze(0)

    # Get the model's prediction
    with torch.no_grad():
        result = model(face_tensor)

    # Flatten the result and convert to a NumPy array
    result = result.flatten().numpy()

    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos