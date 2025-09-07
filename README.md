 ❤️ Heart Disease Analysis & Predictor

This project is a **Heart Disease Analysis and Prediction system** built with **Python, Scikit-learn, and Streamlit**. It allows you to **analyze a heart disease dataset**, train machine learning models, and predict the likelihood of heart disease for new patients.

---
 🛠 Features

**1. Data Analysis**
- Load and explore heart disease dataset.
- Handle missing values using median and most frequent strategies.
- Visualize distributions of numerical and categorical features using **histograms, countplots, and boxplots**.
- Feature correlation heatmap to understand relationships between variables.

** 2. Machine Learning Models**
- Train **Logistic Regression** and **Random Forest** classifiers.
- Evaluate models using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
- Hyperparameter tuning using **RandomizedSearchCV** for the Random Forest model.
- Save trained models and preprocessors for future use (`joblib`).

 **3. Heart Disease Predictor App (Streamlit)**
- Interactive UI to input patient data:
  - Age
  - Sex
  - Chest Pain Type
  - Resting BP
  - Cholesterol
  - Exercise Induced Angina
  - ST Slope
  - Oldpeak (ST Depression)
- Predict **likelihood of heart disease**.
- Shows **results clearly**:  
  - ✅ No Heart Disease  
  - ⚠️ Likely Heart Disease

---

**⚙️ Technologies Used**
- Python 3.x
- Pandas, NumPy – Data manipulation
- Matplotlib, Seaborn – Data visualization
- Scikit-learn – Machine learning models and preprocessing
- Streamlit – Interactive web app interface
- Joblib – Save/load models

---


bash
Copy code
streamlit run app.py
