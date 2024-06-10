from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the classifier and fit it
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Base directory for your image dataset
#base_dir = '/path/to/image/folders'
#features = []
#labels = []

# Extract features from each image in each subdirectory
#for folder_name in os.listdir(base_dir):
#    folder_path = os.path.join(base_dir, folder_name)
#    if os.path.isdir(folder_path):
#        for file in os.listdir(folder_path):
#            if file.endswith(('png', 'jpg', 'jpeg')):
#                file_path = os.path.join(folder_path, file)
#                img_features = extract_features(file_path, base_model)
#                features.append(img_features)
#                # Assign labels: 0 for negative class, 1 for all others
#                labels.append(0 if folder_name == 'negative_class_name' else 1)

# Convert to NumPy arrays for training
#features = np.array(features)
#labels = np.array(labels)