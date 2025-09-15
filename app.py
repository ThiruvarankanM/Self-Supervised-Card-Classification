import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the model architecture (must match training)
class SimpleCNN_Encoder(nn.Module):
    def __init__(self):
        super(SimpleCNN_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x):
        return self.encoder(x)

class FullClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FullClassifier, self).__init__()
        self.encoder = SimpleCNN_Encoder().encoder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output

def load_model(model_path, num_classes):
    model = FullClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image, classes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = torch.argmax(output).item()
    return classes[predicted_idx]

# Streamlit UI
st.title('Card Type Classifier')
st.write('Upload a card image to predict its type.')

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

# Set your class names in the same order as training
default_classes = ['bank_card', 'id_card', 'visiting_card', 'voter_id']
num_classes = len(default_classes)

model_path = 'card_classifier.pth'  # Make sure this file is in the same directory

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write('Predicting...')
    if os.path.exists(model_path):
        model = load_model(model_path, num_classes)
        prediction = predict_image(model, image, default_classes)
        st.success(f'This card is predicted as: {prediction}')
    else:
        st.error(f'Model file {model_path} not found. Please upload the trained model.')
else:
    st.info('Please upload an image file to get a prediction.')
