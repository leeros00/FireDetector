import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

fire_dataset = ImageFolder('fire_dataset')
train_size = int(0.8 * len(fire_dataset))
test_size = len(fire_dataset) - train_size
train_dataset, test_dataset = random_split(fire_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class FireDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*32*32, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train(epochs: int=10, model_name: str='fire_detection_model.pth'):
    model = FireDetectionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, batch {i + 1}, loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    
    torch.save(model.state_dict(), model_name)


def predict(image_path: str, model_path: str='fire_detection_model.pth'):
    # Load the trained model
    model = FireDetectionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the image and preprocess it
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_image = transform(image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        
    return predicted_class


def test(model_path: str='fire_detection_model.pth'):
    # Load the trained model
    model = FireDetectionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(len(images)):
                output = model(images[i].unsqueeze(0))
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities).item()
                if predicted_class == labels[i].item():
                    correct += 1
                total += 1

    # Compute accuracy
    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy:.3f}')
    return accuracy