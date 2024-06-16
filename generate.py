import pandas as pd
from PIL import Image
import mahotas as mh
import numpy as np
import os

# Caminho para o arquivo CSV e diretório das imagens
csv_path = 'classifications.csv'
images_dir = 'dataset/'
output_dir = 'output'
features_csv_path = 'features.csv'

# Carregar dados do CSV
df = pd.read_csv(csv_path)

# Certifique-se de que o diretório de saída existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lista para armazenar características extraídas
features_list = []

# Função para calcular os descritores Haralick
def calculate_haralick_features(image):
    features = mh.features.haralick(image)
    mean_features = features.mean(axis=0)
    return {
        'Entropy': mean_features[8],
        'Homogeneity': mean_features[2],
        'Contrast': mean_features[1]
    }

# Função para determinar a etiqueta binária
def get_label(bethesda_system):
    if bethesda_system == "Negative for intraepithelial lesion":
        return 0
    else:
        return 1

# Processar cada entrada no DataFrame
for index, row in df.iterrows():
    image_path = os.path.join(images_dir, row['image_filename'])
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('L')  # Converter para escala de cinza
        
        # Coordenadas do centro do núcleo da célula
        x, y = row['nucleus_x'], row['nucleus_y']
        
        # Calcular o canto superior esquerdo do recorte
        left = x - 50
        top = y - 50
        
        # Recortar a imagem
        cropped = img.crop((left, top, left + 100, top + 100))
        
        # Criar sub-diretório para a classe, se necessário
        class_dir = os.path.join(output_dir, str(row['bethesda_system']))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Salvar a imagem recortada
        cropped.save(os.path.join(class_dir, f'{index}.png'))
        
        # Calcular os descritores Haralick
        cropped_array = np.array(cropped)
        features = calculate_haralick_features(cropped_array)
        
        # Adicionar informações da imagem e suas características à lista
        features['Image'] = f'{index}.png'
        features['Class'] = row['bethesda_system']
        features['Label'] = get_label(row['bethesda_system'])
        features_list.append(features)

# Criar um DataFrame a partir da lista de características e salvar como CSV
features_df = pd.DataFrame(features_list)
features_df.to_csv(features_csv_path, index=False)

print("Processamento concluído!")
