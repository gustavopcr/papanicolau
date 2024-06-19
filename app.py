import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Menu, filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import numpy as np
import os
from functions import *
import torch
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

# Variavei Globais
filePath = ""
img = ""

def open_image(title, img):
    imagen_array = np.array(img)
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.imshow(imagen_array)
    ax.axis('off')  # Ocultar os eixos
    plt.show()
    # Adicionar a figura à janela Toplevel
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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



def converterTonsDeCinza(img):
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
    return imagem

def histograma(img):
    pimg = np.array(img.convert("RGB"))
    hist, bins = np.histogram(pimg.flatten(), 16, [0, 256])
    hist = hist / np.sum(hist)
    plt.bar(bins[:-1], hist, width=np.diff(bins), color='gray')
    plt.xlabel('Tons de Cinza')
    plt.ylabel('Frequência')
    plt.title('Histograma de Tons de Cinza')
    plt.show()

def color_histogram(img):
    h_bins = 16
    v_bins = 8
    # Load the image
    image = img.convert('HSV')
    image_np = np.array(image)

    # Split into HSV channels
    H, S, V = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]

    # Quantize the H and V channels directly
    H_quantized = (H * (h_bins / 256)).astype(int)
    V_quantized = (V * (v_bins / 256)).astype(int)

    # Compute the 2D histogram
    hist_2d, _, _ = np.histogram2d(H_quantized.flatten(), V_quantized.flatten(), bins=[h_bins, v_bins])
    plt.imshow(hist_2d, interpolation='nearest', origin='lower', aspect='auto')
    plt.title('2D Histogram with 16 H bins and 8 V bins')
    plt.xlabel('Hue bins')
    plt.ylabel('Value bins')
    plt.colorbar()
    plt.show()


def show_result(result):
    # Show popup message with prediction result
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Predic Result", f"A imagem é da classe: {result}")
    root.destroy()  # Destroy the hidden root window after showing the popup

# Criar a janela principal
root = tk.Tk()
root.title("Seletor de Imagem")
root.geometry("800x200")  # Aumentar o tamanho da janela para exibir imagens maiores

button = tk.Button(root, text="Selecionar imagem", command=image_path)
button.pack(pady=10)

# Criar o menu principal
menu_bar = Menu(root)

# Adicionar um menu "Arquivo"
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Abrir Imagem Normal", command=lambda: open_image("Imagem Padrão", img))
file_menu.add_command(label="Abrir Imagem Com os Tons de cinza", command=lambda: open_image("Imagem tom de cinza", converterTonsDeCinza(img)))
file_menu.add_command(label="Abrir Histograma De Tons de Cinza da Imagem", command=lambda: histograma(img))
file_menu.add_command(label="Abrir Histograma HSV da Imagem", command=lambda: color_histogram(img))
file_menu.add_command(label="Haralick", command=lambda: haralick(img))
file_menu.add_command(label="Hu", command=lambda: processar_imagem(img))
file_menu.add_command(label="XGBoost Binario", command=lambda: show_result(process_xgboost_binary(img)))
file_menu.add_command(label="XGBoost Multiclasse", command=lambda: show_result(process_xgboost_multiclass(img)))
file_menu.add_command(label="Efficient Net Binario", command=lambda: show_result(predict_binary(img)))
file_menu.add_command(label="Efficient Net Multiclasse", command=lambda: show_result(predict_multi(img)))
file_menu.add_separator()
file_menu.add_command(label="Sair", command=root.quit)
menu_bar.add_cascade(label="Menu", menu=file_menu)

# Configurar a janela para usar o menu
root.config(menu=menu_bar)

# Label para exibir a imagem
label = tk.Label(root)
label.img_label = None  # Inicializar com None

# Iniciar o loop principal da interface
root.mainloop() 