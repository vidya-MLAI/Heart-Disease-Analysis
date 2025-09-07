import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# 1. Load the dataset
df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\heart_dataset.csv")
print("Dataset loaded successfully.")
print(df.head())
# 2. Basic Info
print("\nDataset Info:")
print(df.info())
print(df.describe())      # Summary stats for numerical columns
print(df['target'].value_counts())  # Class distribution


# 3. Check for missing values
print("\nMissing values:\n", df.isnull().sum())
print("\nRows with missing values:")



print(df.columns.tolist())

num_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
cat_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope', 'target']

# Imputer for numerical columns (using median)
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Imputer for categorical columns (using most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Now df should have no missing values
print(df.isnull().sum())
#Numeric data → distributions & shape
num_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

plt.figure(figsize=(15,10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
# Categorical data → counts & category balance
cat_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

plt.figure(figsize=(15, 12))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 2, i)
    sns.countplot(x=df[col])
    plt.title(f'Countplot of {col}')
plt.tight_layout()
plt.show()
#target variable analysis
sns.countplot(x='target', data=df)
plt.title("Distribution of Heart Disease")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()
#feature sample for numeric
import seaborn as sns
import matplotlib.pyplot as plt

num_features = ['age', 'cholesterol', 'max heart rate', 'resting bp s', 'oldpeak']

for col in num_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f'{col} vs Target')
    plt.show()

#feature sample for nominal
cat_features = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

for col in cat_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='target', data=df)
    plt.title(f'{col} vs Target')
    plt.show()
# matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    features = ['chest pain type', 'ST slope', 'exercise angina', 'oldpeak', 'sex', 'age', 'resting bp s',
                'cholesterol']  # add any other relevant features

    X = df[features]
    y = df['target']

    # Identify categorical and numerical features
    categorical_features = ['sex', 'chest pain type', 'ST slope', 'exercise angina']
    numerical_features = [col for col in features if col not in categorical_features]

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)  # drop='first' to avoid dummy variable trap
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit and transform training data, transform test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Now you can feed X_train_processed and y_train into your ML models

    print("Preprocessing done. Ready for model training!")

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

    # Train the models
    lr_model.fit(X_train_processed, y_train)
    rf_model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred_lr = lr_model.predict(X_test_processed)
    y_pred_rf = rf_model.predict(X_test_processed)

    # Evaluate Logistic Regression
    print("Logistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("Classification Report:\n", classification_report(y_test, y_pred_lr))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

    print("\n" + "=" * 50 + "\n")

    # Evaluate Random Forest
    print("Random Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    # Optional: Feature importance from Random Forest
    importances = rf_model.feature_importances_
    print("\nRandom Forest Feature Importances:\n", importances)

    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Initialize base model
    rf = RandomForestClassifier(random_state=42)

    # Randomized Search with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,  # Number of random combinations to try
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit to training data
    random_search.fit(X_train_processed, y_train)

    # Best model
    best_rf = random_search.best_estimator_

    # Predict and evaluate
    y_pred_best = best_rf.predict(X_test_processed)

    print("Tuned Random Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_best))
    print("Classification Report:\n", classification_report(y_test, y_pred_best))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))