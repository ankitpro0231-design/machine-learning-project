# 🏠 Indian Housing Price Prediction

An end-to-end Machine Learning project that predicts house prices in Indian cities using property, location, and geographical features.

The project covers the complete Machine Learning workflow including data preprocessing, feature engineering, model comparison, cross-validation, hyperparameter tuning, evaluation, feature importance analysis, and model saving for deployment.

---

## 📌 Project Overview

House prices depend on several factors such as property size, number of BHKs, location, floor, furnishing, property age, and accessibility to important locations.

This project uses these factors to predict the estimated price of a house in **₹ Lakhs**.
---

### 🎯 Target Variable

price_lakhs

---

🎯 Objectives
- Predict Indian housing prices using Machine Learning
- Perform data cleaning and preprocessing
- Create meaningful features from existing data
- Compare multiple regression algorithms
- Perform 5-fold cross-validation
- Tune model hyperparameters
- Select the best-performing model
- Evaluate model performance using multiple metrics
- Analyze feature importance
- Save the trained model for future predictions
- Prepare model artifacts for a Streamlit application

---

🧠 Machine Learning Workflow

Housing Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Train / Test Split
       │
       ▼
Feature Engineering
       │
       ▼
Data Preprocessing
       │
       ▼
Model Comparison
       │
       ▼
5-Fold Cross Validation
       │
       ▼
Hyperparameter Tuning
       │
       ▼
Model Evaluation
       │
       ▼
Best Model Selection
       │
       ▼
Train Best Model on Full Dataset
       │
       ▼
Save Model & Results
       │
       ▼
Streamlit Application

---

Features

| Feature | Description |
|---|---|
| `BHK` | Number of bedrooms/hall/kitchen units |
| `area` | Area of the property |
| `floor` | Floor on which the property is located |
| `total_floors` | Total number of floors in the building |
| `age` | Age of the property |
| `city` | City of the property |
| `locality` | Locality/neighborhood |
| `latitude` | Latitude of the property |
| `longitude` | Longitude of the property |
| `distance_to_cbd_km` | Distance from Central Business District |
| `distance_to_metro_km` | Distance from the nearest metro |
| `furnishing` | Furnishing status |
| `price_lakhs` | Target property price in ₹ Lakhs |

---

⚙️ Feature Engineering
The project creates additional features to improve the model's ability to understand relationships within the housing data.
Floor Ratio
floor_ratio = floor / total_floors

Represents the relative position of a property within the building.

Top Floor Indicator
is_top_floor

Indicates whether the property is located on the highest floor.
Ground Floor Indicator
is_ground_floor

Indicates whether the property is located on the ground floor.
Area per BHK
area_per_bhk = area / BHK

Represents the approximate area available per BHK.
Log-transformed Area
log_area = log(1 + area)

Helps represent the relationship between property size and price.
New Property Indicator
is_new

Indicates whether the property is 2 years old or newer.
Metro Proximity
near_metro

Indicates whether the property is within 1 km of a metro station.
City-Locality Combination
city_locality = city + "_" + locality

Combines city and locality to capture location-specific patterns.

---

🧹 Data Preprocessing
Before training, the project performs basic data cleaning:
- Removes duplicate records
- Removes properties with invalid prices
- Removes properties with invalid area values
- Removes records where floor > total_floors
The dataset is divided into:
80% → Training Data
20% → Testing Data

The train-test split is stratified by city so that the different cities are represented in both training and testing datasets.

---

🤖 Machine Learning Models
The project compares four regression algorithms.
1. Ridge Regression
A regularized linear regression algorithm that helps reduce overfitting and handles correlated features.
2. Random Forest Regressor
An ensemble learning algorithm that combines multiple decision trees to make predictions.
3. Gradient Boosting Regressor
A boosting algorithm that builds models sequentially, with each model improving upon the errors of previous models.
4. HistGradientBoosting Regressor
A histogram-based implementation of gradient boosting designed for efficient model training.

---

🔬 Model Selection & Hyperparameter Tuning
The baseline models are evaluated using:
5-Fold Cross Validation

The project then performs hyperparameter tuning on:
- Ridge Regression
- Gradient Boosting Regression
using:
RandomizedSearchCV

Ridge Hyperparameter
alpha

Gradient Boosting Hyperparameters
n_estimators
learning_rate
max_depth
min_samples_leaf
subsample

The best configuration is selected using cross-validation based on Mean Absolute Error (MAE).

---

📏 Model Evaluation
The project uses four regression metrics.
MAE — Mean Absolute Error
Measures the average absolute difference between actual and predicted prices.
Lower MAE = Better

RMSE — Root Mean Squared Error
Measures prediction error while giving more importance to larger errors.
Lower RMSE = Better

R² Score
Measures how well the model explains the variation in house prices.
Closer to 1 = Better

MAPE — Mean Absolute Percentage Error
Measures the average prediction error as a percentage.
Lower MAPE = Better

📈 Model Visualization
The training pipeline generates two important visualizations.
Actual vs Predicted
models/actual_vs_pred.png

This plot compares actual house prices with prices predicted by the selected model.

Predictions closer to the diagonal reference line indicate better performance.
Feature Importance
models/feature_importance.png

The project uses Permutation Importance to understand the contribution of the original input features to model performance.

---
💾 Generated Model Files
After running the training script, the following files are generated:
models/
│
├── best_model.joblib
├── metrics.json
├── meta.json
├── actual_vs_pred.png
└── feature_importance.png

best_model.joblib
Contains the final trained Machine Learning pipeline.
metrics.json
Stores:
- Cross-validation results
- Test-set performance
- Selected model
- Best hyperparameters
- Permutation importance
meta.json
Stores metadata used by the Streamlit application, including:
- Available cities
- Localities
- Location statistics
- Furnishing categories
actual_vs_pred.png
Actual vs predicted price visualization.
feature_importance.png
Permutation feature importance visualization.

---

📁 Project Structure

indian-housing-price-prediction/
│
├── data/
│   └── indian_housing_synthetic_1k.csv
│
├── models/
│   ├── best_model.joblib
│   ├── metrics.json
│   ├── meta.json
│   ├── actual_vs_pred.png
│   └── feature_importance.png
│
├── train.py
├── app.py
├── requirements.txt
└── README.md

---

🛠️ Tech Stack
Programming Language
- Python
Data Processing
- Pandas
- NumPy
Machine Learning
- Scikit-learn
Model Persistence
- Joblib
Data Visualization
- Matplotlib
Deployment / Application
- Streamlit
---

🔄 Prediction Workflow

User enters property details
          │
          ▼
Feature Engineering
          │
          ▼
Data Preprocessing
          │
          ▼
Trained ML Model
          │
          ▼
Price Prediction
          │
          ▼
Estimated Price in ₹ Lakhs

---

🔐 Data Leakage Prevention
Feature engineering and preprocessing are implemented as part of the Machine Learning pipeline.

The custom FeatureEngineer transformer creates derived features without using the target variable, helping prevent target leakage.

The project also applies a logarithmic transformation to the target variable because house prices are right-skewed.

---

🔁 Cross Validation
The project uses:
5-Fold Cross Validation

with shuffled folds and a fixed random state.

This provides a more reliable estimate of model performance than relying on a single training split.

---

🎲 Reproducibility
A fixed random seed is used throughout the training process:
RANDOM_STATE = 42


This helps make the training and evaluation process reproducible.

---

✨ Key Features
- 🏠 Indian housing price prediction
- 📊 Data preprocessing
- ⚙️ Automated feature engineering
- 🔢 Numerical feature preprocessing
- 🏷️ Categorical feature encoding
- 🔄 5-fold cross-validation
- 🤖 Multiple regression algorithms
- 🔬 Hyperparameter tuning
- 📏 Multiple evaluation metrics
- 📈 Actual vs predicted visualization
- 💡 Permutation feature importance
- 💾 Model persistence using Joblib
- 🌐 Streamlit-ready model artifacts

---

🔮 Future Improvements
- Use real-world housing datasets
- Add more cities and localities
- Include property type and amenities
- Add historical real-estate price trends
- Add additional geographical features
- Compare more advanced boosting algorithms
- Add SHAP-based model explainability
- Add automated model retraining
- Deploy the application publicly
- Build a REST API using FastAPI
- Add model monitoring and performance tracking

---

⭐ Project Summary
Indian Housing Price Prediction is an end-to-end Machine Learning regression project that predicts residential property prices using property characteristics, location information, and geographical features.

The project demonstrates a complete Machine Learning workflow from data preprocessing and feature engineering to model selection, hyperparameter tuning, evaluation, interpretability, and deployment preparation.
```text
