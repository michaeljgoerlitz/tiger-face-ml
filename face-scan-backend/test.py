import torch
from two import extract_features, recommend, get_file_dict
from mtcnn import MTCNN
import pickle


model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Get model ready for feature extraction, removing last layer
model = torch.nn.Sequential(*(list(model.children())[:-1]))

detector = MTCNN()

file_dict = get_file_dict('all_tigers')

feature_list = pickle.load(open('embedding.pkl', 'rb'))

features = extract_features('Screenshot 2024-01-15 at 11.03.56 AM.jpg', model, detector)
index_pos = recommend(feature_list, features)
print("done, ", index_pos)
print(file_dict[index_pos])