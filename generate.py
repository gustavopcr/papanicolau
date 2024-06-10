import pandas as pd
from PIL import Image
import os

# Caminho para o arquivo CSV e diretório das imagens
csv_path = 'classifications.csv'
images_dir = 'dataset/'
output_dir = 'output'

# Carregar dados do CSV
df = pd.read_csv(csv_path)

# Certifique-se de que o diretório de saída existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Processar cada entrada no DataFrame
for index, row in df.iterrows():
    image_path = os.path.join(images_dir, row['image_filename'])
    if os.path.exists(image_path):
        img = Image.open(image_path)
        
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

print("Processamento concluído!")
