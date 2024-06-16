import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Menu, filedialog
from PIL import Image, ImageTk # type: ignore
from matplotlib import pyplot as plt
import numpy as np
import os
from functions import *

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
    img_tk = ImageTk.PhotoImage(imagem)
    if label.img_label:
        label.img_label.destroy()
    label.img_label = tk.Label(root, image=img_tk)
    label.img_label.image = img_tk  # Manter uma referência da imagem
    label.img_label.pack()

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



def open_image(img):
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
file_menu.add_command(label="Abrir Imagem Normal", command=lambda: open_image(img))
file_menu.add_command(label="Abrir Imagem Com os Tons de cinza", command=lambda: converterTonsDeCinza(img))
file_menu.add_command(label="Abrir Histograma De Tons de Cinza da Imagem", command=lambda: histograma(img))
file_menu.add_command(label="Abrir Histograma HSV da Imagem", command=lambda: color_histogram(img))
file_menu.add_command(label="Haralick", command=lambda: haralick(img))
file_menu.add_command(label="Hu", command=lambda: processar_imagem(img))
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