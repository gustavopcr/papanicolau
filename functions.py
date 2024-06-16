import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Menu, filedialog
from PIL import Image, ImageTk # type: ignore
from matplotlib import pyplot as plt
import numpy as np
import os
import mahotas as mh
from skimage.measure import moments, moments_hu
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy

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

def hu_moments(image):
    m = moments(image)
    # Calcular os momentos invariantes de Hu
    huMoments = moments_hu(m)
    # Aplicar log transform para trazer os valores para uma escala mais fácil de manipular
    for i in range(0, 7):
        huMoments[i] = -np.sign(huMoments[i]) * np.log10(abs(huMoments[i]))
    return huMoments

def processar_imagem(img):
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

    print("hu_gray: ", hu_gray)
    print("hu_h: ", hu_h)
    print("hu_s: ", hu_s)
    print("hu_v: ", hu_v)

    return hu_gray, hu_h, hu_s, hu_v
    
def normalize_features(features):
    # Normalize the features for better visualization
    normalized = (features - np.min(features)) / (np.max(features) - np.min(features))
    return normalized
