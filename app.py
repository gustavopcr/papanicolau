import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Menu, filedialog
from PIL import Image, ImageTk # type: ignore
from matplotlib import pyplot as plt
import numpy as np
import os
import mahotas as mh
from sklearn.preprocessing import MinMaxScaler
import cv2

# Variavei Globais
filePath = ""
img = ""


def say_hello():
    print("Hello!")

def image_path():
    # Abrir a caixa de diálogo para selecionar o arquivo de imagem
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        # Abrir e renderizar a imagem
        global img
        img = Image.open(file_path)
        # Salva a caminho para a imagem
        global filePath
        filePath = file_path
        formatName = os.path.basename(filePath)
        textRender = f"-- Imagem selecionada --\n '{formatName}' \n para mais opções selecione o menu"

        if hasattr(label, 'file_selected'):
            label.file_selected.destroy()
        if label.img_label:
            label.img_label.destroy()
        label.file_selected = tk.Label(root, text=textRender,  padx=10, pady=5, font=("Arial", 10), justify="center")
        label.file_selected.pack()


def converterTonsDeCinza():
    global img
    if img:
        largura, altura = img.size
        imagem = img.convert('RGB')
        # Obter os pixels da imagem
        pixels = imagem.load()

        # Percorrer todos os pixels da imagem
        for x in range(largura):
            for y in range(altura):
                # Obter os valores dos canais de cor RGB para o pixel atual
                r, g, b = pixels[x, y]

                # Calcular o novo valor de intensidade Y usando a fórmula
                y_novo = int(0.299 * r + 0.587 * g + 0.114 * b)

                # Atualizar o pixel na imagem com o novo valor de intensidade Y
                pixels[x, y] = (y_novo, y_novo, y_novo)
        img_tk = ImageTk.PhotoImage(imagem)
        if label.img_label:
            label.img_label.destroy()
        label.img_label = tk.Label(root, image=img_tk)
        label.img_label.image = img_tk  # Manter uma referência da imagem
        label.img_label.pack()

def histograma():
    global img
    if img:
        pimg = np.array(img.convert("RGB"))
        hist, bins = np.histogram(pimg.flatten(), 16, [0, 256])
        hist = hist / np.sum(hist)
        plt.bar(bins[:-1], hist, width=np.diff(bins), color='gray')
        plt.xlabel('Tons de Cinza')
        plt.ylabel('Frequência')
        plt.title('Histograma de Tons de Cinza')
        plt.show()
def haralick():
        # Read image using mahotas
    img = mh.imread(filePath)
    
    # Check if the image is already in grayscale; if not, convert it
    if img.ndim == 3:
        # Convert image to grayscale
        img = mh.colors.rgb2grey(img)
    
    # Ensure the image is in the correct format (integer type)
    img = img.astype(np.uint8)
    
    # Calculate Haralick features
    haralick_features = mh.features.haralick(img).mean(axis=0)
    
    # Feature names
    feature_names = [
        "Angular Second Moment",
        "Contrast",
        "Correlation",
        "Variance",
        "Inverse Difference Moment",
        "Sum Average",
        "Sum Variance",
        "Sum Entropy",
        "Entropy",
        "Difference Variance",
        "Difference Entropy",
        "Information Measure of Correlation 1",
        "Information Measure of Correlation 2"
    ]
    
    # Normalize features for better visualization
    normalized_features = normalize_features(haralick_features)
    
    # Plot the features
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(normalized_features)), normalized_features, tick_label=feature_names)
    plt.xlabel('Haralick Feature')
    plt.ylabel('Value')
    plt.title('Haralick Texture Features')
    plt.xticks(rotation=90)
    plt.show()

def normalize_features(features):
    # Normalize the features for better visualization
    normalized = (features - np.min(features)) / (np.max(features) - np.min(features))
    return normalized


def hu_moments():
    # Read the input image
    image = cv2.imread(filePath)

    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply thresholding on gray image
    ret,thresh = cv2.threshold(gray,150,255,0)

    # Find the contours in the image
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("Number of Contours detected:",len(contours))

    # Find the moments of first contour
    cnt = contours[0]
    M = cv2.moments(cnt)
    Hm = cv2.HuMoments(M)

    # Draw the contour
    cv2.drawContours(image, [cnt], -1, (0,255,255), 3)
    x1, y1 = cnt[0,0]
    cv2.putText(image, 'Contour:1', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # print the moments of the first contour
    print("Hu-Moments of first contour:\n", Hm)
    cv2.imshow("Hu-Moments", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_image():
    global img
    if img:
        img_tk = ImageTk.PhotoImage(img)
        

        # Se a imagem já estiver sendo exibida, remova-a antes de adicionar a nova
        if label.img_label:
            label.img_label.destroy()

        # Criar um label para exibir a imagem
        label.img_label = tk.Label(root, image=img_tk)
        label.img_label.image = img_tk  # Manter uma referência da imagem
        label.img_label.pack()

# Criar a janela principal
root = tk.Tk()
root.title("Seletor de Imagem")
# root.geometry("800x200")  # Aumentar o tamanho da janela para exibir imagens maiores

button = tk.Button(root, text="Selecionar imagem", command=image_path)
button.pack(pady=10)

# Criar o menu principal
menu_bar = Menu(root)

# Adicionar um menu "Arquivo"
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Abrir Imagem Normal", command=open_image)
file_menu.add_command(label="Abrir Imagem Com os Tons de cinza", command=converterTonsDeCinza)
file_menu.add_command(label="Abrir Histograma da Imagem", command=histograma)
file_menu.add_command(label="Haralick", command=haralick)
file_menu.add_command(label="HuMoments", command=hu_moments)
file_menu.add_separator()
file_menu.add_command(label="Sair", command=root.quit)
menu_bar.add_cascade(label="Menu", menu=file_menu)

# Adicionar um menu "Editar"
edit_menu = Menu(menu_bar, tearoff=0)
edit_menu.add_command(label="Desfazer", command=say_hello)
edit_menu.add_command(label="Refazer", command=say_hello)
menu_bar.add_cascade(label="Editar", menu=edit_menu)

# Configurar a janela para usar o menu
root.config(menu=menu_bar)

# Label para exibir a imagem
label = tk.Label(root)
label.img_label = None  # Inicializar com None

# Iniciar o loop principal da interface
root.mainloop() 