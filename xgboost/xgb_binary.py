import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the CSV file into a DataFrame
data = pd.read_csv('csv/haralick_binary.csv')

# Separate features (X) and labels (y)
X = data.drop(['filename', 'label'], axis=1)
y = data['label']

# Get feature names
feature_names = X.columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the model to a file
joblib_file = "xgboost/xgb_binary_model.pkl"
joblib.dump(model, joblib_file)
print(f'Model saved to {joblib_file}')
