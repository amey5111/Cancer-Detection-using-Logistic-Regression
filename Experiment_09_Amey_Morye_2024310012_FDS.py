import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load dataset
data = pd.read_csv('survey lung cancer.csv')

# Check for missing values and handle them
# Select only numeric columns for calculating the mean
numeric_data = data.select_dtypes(include=['number'])

# Fill missing values in numeric columns with the mean of those columns
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())
# Split into features and target
X = data.drop(columns=['LUNG_CANCER'])
y = data['LUNG_CANCER']
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns

# Identify numerical features
numerical_features = X.select_dtypes(exclude=['object']).columns

# Create a ColumnTransformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)]) # handle_unknown='ignore' to avoid errors with unseen values during prediction

# Fit and transform the data
X_scaled = preprocessor.fit_transform(X)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
# Test model
y_pred = model.predict(X_test)
# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
import joblib
joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved as 'logistic_regression_model.pkl'")


import mlflow
import mlflow.sklearn

mlflow.set_experiment("Cancer Detection with Logistic Regression")

with mlflow.start_run():
  mlflow.log_param("model_type", "Logistic Regression")
  mlflow.log_metric("accuracy", accuracy)
  mlflow.sklearn.log_model(model, "model")
  print("Model logged in MLflow.")
