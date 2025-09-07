import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import streamlit as st


# 1. Load the dataset
df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\heart_dataset.csv")
print("Dataset loaded successfully.")

# 2. Basic Info
print(df.info())
print(df.describe())
print(df['target'].value_counts())

# 3. Handle missing values
num_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
cat_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope', 'target']

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

for col in ['sex', 'chest pain type', 'ST slope', 'exercise angina']:
    print(f"{col}: {df[col].unique()}")

# 4. Visualizations
plt.figure(figsize=(15,10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 12))
for i, col in enumerate(cat_cols[:-1], 1):
    plt.subplot(3, 2, i)
    sns.countplot(x=df[col])
    plt.title(f'Countplot of {col}')
plt.tight_layout()
plt.show()

sns.countplot(x='target', data=df)
plt.title("Distribution of Heart Disease")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Feature selection
features = ['chest pain type', 'ST slope', 'exercise angina', 'oldpeak', 'sex', 'age', 'resting bp s', 'cholesterol']
X = df[features]
y = df['target']

categorical_features = ['sex', 'chest pain type', 'ST slope', 'exercise angina']
numerical_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing done. Ready for model training!")

# Train models
lr_model = LogisticRegression(random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
lr_model.fit(X_train_processed, y_train)
rf_model.fit(X_train_processed, y_train)

y_pred_lr = lr_model.predict(X_test_processed)
y_pred_rf = rf_model.predict(X_test_processed)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=30,
                                   cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train_processed, y_train)

best_rf = random_search.best_estimator_
y_pred_best = best_rf.predict(X_test_processed)

print("\nTuned Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

# Save model and scaler
joblib.dump(best_rf, "tuned_random_forest_model.pkl")
joblib.dump(preprocessor, "Preprocessor.pkl")
print("\nModel and preprocessor saved successfully.")

#testing
# Load model and preprocessor
model = joblib.load("tuned_random_forest_model.pkl")
preprocessor = joblib.load("Preprocessor.pkl")

# Example new data (must match feature structure)
new_data = pd.DataFrame([{
    'chest pain type': 4,         # Typical angina — more associated with disease
    'ST slope': 2,           # 'flat' slope is often linked to heart issues
    'exercise angina': 1,         # 1 = yes → presence of exercise-induced angina
    'oldpeak': 3.0,               # Higher values → more ST depression → riskier
    'sex': 1,                     # 1 = male (males have higher risk overall)
    'age': 65,                    # Older age increases risk
    'resting bp s': 170,          # Elevated systolic BP
    'cholesterol': 290            # High cholesterol level
}])

# Preprocess and predict
new_data_processed = preprocessor.transform(new_data)
prediction = model.predict(new_data_processed)
print("Prediction:", prediction[0])  # 0 = No disease, 1 = Disease

import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("tuned_random_forest_model.pkl")
preprocessor = joblib.load("Preprocessor.pkl")

st.title("❤️ Heart Disease Predictor")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)])
cp_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
rest_bp = st.number_input("Resting BP (Systolic)", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
ex_angina = st.selectbox("Exercise Induced Angina", [("No", 0), ("Yes", 1)])
st_slope = st.selectbox("ST Slope", [("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

if st.button("Predict"):
    data = pd.DataFrame([{
        'age': age,
        'sex': sex[1],
        'chest pain type': cp_type,
        'resting bp s': rest_bp,
        'cholesterol': chol,
        'exercise angina': ex_angina[1],
        'ST slope': st_slope[1],
        'oldpeak': oldpeak
    }])

    processed = preprocessor.transform(data)
    result = model.predict(processed)
    if result[0] == 1:
        st.error("⚠️ Likely Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

