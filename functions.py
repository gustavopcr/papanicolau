import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Menu, filedialog
from PIL import Image, ImageTk # type: ignore
from matplotlib import pyplot as plt
import numpy as np
import os
from skimage.measure import moments, moments_hu
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from torchvision import transforms
import joblib
import xgboost as xgb
import pandas as pd


binary_pred = ['negative', 'positive']
multiclass_pred = ['negative', 'ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC']


# Caminho dos arquivos de modelo
xgboost_binary_path = "xgboost/xgboost_binary_model.pkl"
xgboost_multi_path = "xgboost/xgboost_multiclass_model.pkl"

# Inicializando variáveis de modelo
xgboost_binary_model = None
xgboost_multi_model = None

# Tentando carregar o modelo binário
try:
    if os.path.exists(xgboost_binary_path):
        xgboost_binary_model = joblib.load(xgboost_binary_path)
        print(f'Model loaded from {xgboost_binary_path}')
        feature_names = xgboost_binary_model.feature_names_in_
    else:
        print(f'Model file not found: {xgboost_binary_path}')
except Exception as e:
    print(f'Error loading binary model: {e}')

# Tentando carregar o modelo multiclasse
try:
    if os.path.exists(xgboost_multi_path):
        xgboost_multi_model = joblib.load(xgboost_multi_path)
        print(f'Model loaded from {xgboost_multi_path}')
    else:
        print(f'Model file not found: {xgboost_multi_path}')
except Exception as e:
    print(f'Error loading multiclass model: {e}')

# Verificando se os modelos foram carregados corretamente
if xgboost_binary_model is None:
    print('Binary model not loaded.')
if xgboost_multi_model is None:
    print('Multiclass model not loaded.')


def process_xgboost_binary(img):
    image = img.convert('L')  # Convert to grayscale
    image = image.resize((64, 64))  # Resize to 64x64, assuming your model was trained on this size
    image_df = pd.DataFrame([haralick(img)], columns=feature_names)
    image_df = image_df[feature_names]
    prediction = xgboost_binary_model.predict(image_df)
    prediction = int(prediction[0])
    return binary_pred[prediction]

def process_xgboost_multiclass(img):
    image = img.convert('L')  # Convert to grayscale
    image = image.resize((64, 64))  # Resize to 100x100, assuming your model was trained on this size
    image_df = pd.DataFrame([haralick(img)], columns=feature_names)
    image_df = image_df[feature_names]
    prediction_proba = xgboost_multi_model.predict_proba(image_df.values.reshape(1, -1))
    predicted_class = prediction_proba.argmax(axis=1)[0]
    print('prediction: ' , predicted_class)
    return multiclass_pred[predicted_class]


# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class EfficientNetWithHaralickB(nn.Module):
    def __init__(self):
        super(EfficientNetWithHaralickB, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, 128)  # Intermediate layer
        self.fc1 = nn.Linear(128 + 18, 64)  # 128 from EfficientNet, 18 from Haralick features
        self.fc2 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x, haralick_features):
        x = self.efficientnet(x)
        x = torch.cat((x, haralick_features), dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
    
class EfficientNetWithHaralickM(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithHaralickM, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, 128)  # Intermediate layer
        self.fc1 = nn.Linear(128 + 18, 64)  # 128 from EfficientNet, 18 from Haralick features
        self.fc2 = nn.Linear(64, num_classes)  # Output layer with num_classes units

    def forward(self, x, haralick_features):
        x = self.efficientnet(x)
        x = torch.cat((x, haralick_features), dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
# Certifique-se de que os pesos da rede existem
effnet_binary_path = 'efficient_net/efficientnet_binary.pth'
effnet_multi_path = 'efficient_net/efficientnet_multiclass.pth'




if not (os.path.exists(effnet_binary_path) and os.path.exists(effnet_multi_path)):
    print("Os pesos da rede não foram encontrados.")
    exit()

model_binary = EfficientNetWithHaralickB()

# Load the saved model weights
model_binary.load_state_dict(torch.load(effnet_binary_path,  map_location=torch.device('cpu')))
# Set the model to evaluation mode
model_binary.eval()

model_multi = EfficientNetWithHaralickM(6)

# Load the saved model weights
model_multi.load_state_dict(torch.load(effnet_multi_path,  map_location=torch.device('cpu')))
# Set the model to evaluation mode
model_multi.eval()



def predict_binary(img):
    # Load and preprocess the image
    image = img.convert('RGB')
    image = transform(image).unsqueeze(0).to('cpu')  # Add batch dimension and move to device

    # Convert Haralick features to tensor
    haralick_features = torch.tensor(haralick(img.convert('RGB')), dtype=torch.float32).unsqueeze(0).to('cpu')  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        output = model_binary(image, haralick_features)
        _, predicted = torch.max(output, 1)
    print('predicted.item()', predicted.item())

    return binary_pred[predicted.item()]

def predict_multi(img):
    # Load and preprocess the image
    image = img.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Extract Haralick features
    haralick_features_tensor = torch.tensor(haralick(image), dtype=torch.float32).unsqueeze(0)
    # Make prediction
    with torch.no_grad():
        output = model_multi(image_tensor, haralick_features_tensor)
        _, predicted = torch.max(output, 1)

    return multiclass_pred[predicted.item()]




def matriz_coocorrencia(img):
    image = np.array(img.convert("L"))
    levels = 16
    distances = [1, 2, 4, 8, 16, 32]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    img_gray = (image / 256 * levels).astype(np.uint8)
    glcm_list = []

    for distance in distances:
        glcm = graycomatrix(img_gray, distances=[distance], angles=angles, levels=levels, symmetric=True, normed=True)
        glcm_list.append(glcm)

    return glcm_list


def plot_matriz_coocorrencia(img):
    # Calcular as matrizes de co-ocorrência
    glcm_list = matriz_coocorrencia(img)
    distances = [1, 2, 4, 8, 16, 32]
    # Plotar as matrizes
    plt.figure(figsize=(15, 10))
    for i, glcm in enumerate(glcm_list, 1):
        plt.subplot(2, 3, i)
        plt.imshow(glcm[:, :, 0, 0], cmap='gray', interpolation='nearest')
        plt.title(f"Distância {distances[i-1]}")
        plt.axis('off')

    plt.show()

def calculate_entropy(glcm):
    glcm_normalized = glcm / np.sum(glcm, axis=(0, 1))
    entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + (glcm_normalized == 0)), axis=(0, 1))
    return entropy

def haralick(img):
    # Properties to compute
    properties = ['contrast', 'homogeneity']
    glcm_list = matriz_coocorrencia(img)
    features = []

    for glcm in glcm_list:
        glcm_features = {}
        for prop in properties:
            descriptor = graycoprops(glcm, prop)
            glcm_features[prop] = descriptor.mean()
        
        entropy = calculate_entropy(glcm)
        glcm_features['entropy'] = entropy.mean()
        
        features.extend(glcm_features.values())
    return features

def plot_haralick(img):
    properties = ['Contrast', 'Homogeneity', 'Entropy']
    distances = [1, 2, 4, 8, 16, 32]
    features = haralick(img)

    plt.figure(figsize=(15, 5))

    for i, prop in enumerate(properties):
        start_idx = i * len(distances)
        end_idx = start_idx + len(distances)
        prop_values = features[start_idx:end_idx]
        plt.plot(range(1, 7), prop_values, marker='o', label=prop)

    plt.title('Haralick Features')
    plt.xlabel('Distance Index')
    plt.ylabel('Value')
    plt.xticks(range(1, 7), distances)
    plt.legend()
    plt.grid(True)
    plt.show()

def hu_moments(image):
    m = moments(image)
    # Calcular os momentos invariantes de Hu
    huMoments = moments_hu(m)
    # Aplicar log transform para trazer os valores para uma escala mais fácil de manipular
    for i in range(0, 7):
        huMoments[i] = -np.sign(huMoments[i]) * np.log10(abs(huMoments[i]))
    return huMoments

def processar_imagem_hu(img):
    gray_image = np.array(img.convert("L"))
    hsv_image = np.array(img.convert("HSV"))
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]
    hu_gray = hu_moments(gray_image)
    
    # Calcular momentos invariantes de Hu para cada canal do HSV
    hu_h = hu_moments(h_channel)
    hu_s = hu_moments(s_channel)
    hu_v = hu_moments(v_channel)

    return hu_gray, hu_h, hu_s, hu_v

def plot_hu_moments(img):
    hu_gray, hu_h, hu_s, hu_v = processar_imagem_hu(img)
    labels = ['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7']
    channels = ['Gray', 'H', 'S', 'V']
    hu_values = [hu_gray, hu_h, hu_s, hu_v]

    plt.figure(figsize=(10, 6))

    for i, channel in enumerate(channels):
        plt.plot(labels, hu_values[i], marker='o', label=channel)

    plt.title('Hu Moments')
    plt.xlabel('Hu Moment')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def normalize_features(features):
    # Normalize the features for better visualization
    normalized = (features - np.min(features)) / (np.max(features) - np.min(features))
    return normalized
