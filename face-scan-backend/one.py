import torch
import torchvision.models as models
import os
import pickle
from torchvision import transforms
from PIL import Image

# Load pre-trained model such as ResNet50
# model = models.resnet50(pretrained=True)
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Modify the model for feature extraction, removing the last layer
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Pre-process the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

folder_path = './all_tigers'

file_to_num = {}
curr_file = 0

all_features = []

# Loop through the files in the folder
for file_name in os.listdir(folder_path):
# for file_name in tqdm(os.listdir(folder_path)):
  file_path = os.path.join(folder_path, file_name)
  # print(f"Processing file: {file_path}")
  image = Image.open(file_path)
  image = transform(image).unsqueeze(0)  # Add batch dimension

  with torch.no_grad():
    features = model(image)
    features = features.squeeze().numpy()
    all_features.append(features)

  file_to_num[curr_file] = file_path
  curr_file += 1

pickle.dump(all_features, open('embedding.pkl','wb'))