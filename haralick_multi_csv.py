import os
import csv
from PIL import Image
from functions import haralick

def get_label(class_name):
    class_labels = {
        "Negative for intraepithelial lesion": 0,
        "ASC-US": 1,
        "ASC-H": 2,
        "LSIL": 3,
        "HSIL": 4,
        "SCC": 5
    }
    return class_labels.get(class_name, -1)  # Return -1 if class name is not found

def main():
    output_folder = 'output'
    output_csv = 'haralick_multi.csv'

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename']
        for idx in range(6):
            header.extend([f'entropy_{idx}', f'contrast_{idx}', f'homogeneity_{idx}'])
        header.append('label')
        writer.writerow(header)

        for folder in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder)
            if os.path.isdir(folder_path):
                label = get_label(folder)
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(folder_path, filename)
                        img = Image.open(img_path)
                        haralick_features = haralick(img)
                        row = []
                        for idx in range(6):
                            start = idx * 3
                            end = (idx + 1) * 3
                            row.extend(haralick_features[start:end])
                        row.append(label)
                        # Prepend directory name to filename for uniqueness
                        writer.writerow([os.path.join(folder, filename)] + row)

if __name__ == "__main__":
    main()
