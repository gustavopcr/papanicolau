import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import joblib
from scipy.stats import randint, uniform

# Load the CSV file into a DataFrame
data = pd.read_csv('csv/haralick_binary.csv')

# Separate features (X) and labels (y)
X = data.drop(['filename', 'label'], axis=1)
y = data['label']

# Get feature names
feature_names = X.columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBClassifier()

# Define the parameter distributions for RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(50, 201),  # Range from 50 to 200
    'learning_rate': uniform(0.01, 0.2),  # Continuous range between 0.01 and 0.21
    'max_depth': randint(3, 8),  # Range from 3 to 7
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=100,
                                   cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Perform RandomizedSearchCV
random_search.fit(X_train, y_train)

# Print the accuracy for each model
print("Accuracy for each model:")
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
for mean_score, std_score, params in zip(means, stds, random_search.cv_results_['params']):
    print(f"Mean Accuracy: {mean_score:.4f} (±{std_score:.4f}) for {params}")

# Get the best model and its parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_

print(f'\nBest parameters found: {best_params}')

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the best model to a file
joblib_file = "xgboost/xgboost_binary_model.pkl"
joblib.dump(best_model, joblib_file)
print(f'Best model saved to {joblib_file}')
