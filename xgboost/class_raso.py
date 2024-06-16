import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Load and process images
def load_and_process_image(image_path):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((100, 100))  # Resize to 100x100 if not already
        img_array = np.array(img).flatten()  # Flatten the image to a 1D array
        return img_array
    return None

# Prepare the data
csv_path = 'classifications.csv'
images_dir = 'dataset/'
df = pd.read_csv(csv_path)
train_indices, test_indices = train_test_split(df.index, train_size=0.8, random_state=None)

X_train = []
y_train = []
X_test = []
y_test = []

for index in train_indices:
    row = df.loc[index]
    image_path = os.path.join(images_dir, row['image_filename'])
    img_array = load_and_process_image(image_path)
    if img_array is not None:
        X_train.append(img_array)
        y_train.append(row['bethesda_system'])

for index in test_indices:
    row = df.loc[index]
    image_path = os.path.join(images_dir, row['image_filename'])
    img_array = load_and_process_image(image_path)
    if img_array is not None:
        X_test.append(img_array)
        y_test.append(row['bethesda_system'])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"XGBoost Test Accuracy: {accuracy}")
print(f"XGBoost Test Precision: {precision}")
print(f"XGBoost Test Recall: {recall}")
print(f"XGBoost Test F1-Score: {f1}")
print(f"XGBoost Test ROC AUC: {roc_auc}")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters for XGBoost: ", grid_search.best_params_)
best_xgb_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Best XGBoost Test Accuracy: {accuracy}")
print(f"Best XGBoost Test Precision: {precision}")
print(f"Best XGBoost Test Recall: {recall}")
print(f"Best XGBoost Test F1-Score: {f1}")
print(f"Best XGBoost Test ROC AUC: {roc_auc}")
