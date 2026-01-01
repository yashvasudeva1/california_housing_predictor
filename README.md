# California Housing Price Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b)
![Scikit-learn](https://img.shields.io/badge/Model-Random%20Forest-orange)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-3f4f75)

An interactive Machine Learning web application built with **Streamlit** to predict median housing prices in California. This dashboard uses a **Random Forest Regressor** to analyze census data and provides deep insights into feature importance, model performance, and geospatial trends.

## Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Author](#-author)

## Overview

The California Housing Price Predictor is an end-to-end data science project that transforms raw census data into actionable insights. It allows users to simulate housing scenarios (e.g., "How does house age affect price?") and visualizing the results instantly. The application is designed for both technical users who want to analyze residuals and non-technical users who want simple price estimates.

## Key Features

### 1. Interactive Data Exploration
- **Statistical Summary:** Instant view of mean, median, and standard deviation for all features.
- **Distribution Analysis:** Switch between Histograms and Box Plots to detect outliers.
- **Geospatial Mapping:** Interactive map visualizing high-value districts using Latitude/Longitude.

### 2. Real-Time Predictions
- **Manual Simulation:** Adjust sliders for Income, Rooms, and Age to get a custom price estimate.
- **Confidence Intervals:** Provides a 95% confidence range based on individual tree variance.
- **Dataset Sampling:** Pick random real-world examples to test the model's accuracy.

### 3. Advanced Model Analysis
- **Performance Metrics:** Live calculation of $R^2$, RMSE, and MAE on test data.
- **Residual Analysis:** Visualize error distribution to check for model bias.
- **Feature Importance:** See exactly which factors (e.g., Median Income, Location) drive prices the most.

## Dataset

The model is trained on the **California Housing Dataset**. Key inputs include:

| Feature | Description |
| :--- | :--- |
| **MedInc** | Median income in block group (in tens of thousands) |
| **HouseAge** | Median house age in block group |
| **AveRooms** | Average number of rooms per household |
| **AveBedrms** | Average number of bedrooms per household |
| **Population** | Block group population |
| **AveOccup** | Average number of household members |
| **Latitude/Longitude** | Geographic coordinates |

**Target Variable:** Median House Value (in $100,000s).

## Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest)
- **Visualization:** Plotly Express, Plotly Graph Objects
- **Deployment:** Streamlit Cloud / Local

## Installation

**1. Clone the Repository**
```bash
git clone [https://github.com/yashvasudeva1/california_housing_predictor.git](https://github.com/yashvasudeva1/california_housing_predictor.git)
cd california_housing_predictor
```
**2. Install Dependencies**

```Bash

pip install -r requirements.txt
```

**3. Verify Data Files Ensure these files are present in the root directory:**
```
California_Housing.csv

random_forest_model.pkl
```

**4. Run the App**

```Bash

streamlit run app.py
Usage
Home: Review dataset statistics and raw data.

Data Exploration: Use the tabs to view Correlation Heatmaps and Geographic plots.

Make Predictions: Select "Manual Input" to create a hypothetical house and see the predicted price.

Feature Analysis: Explore the "Impact Analysis" chart to see how sensitive prices are to specific changes (e.g., changing Income from 3.0 to 5.0).
```
Model Architecture
Algorithm: Random Forest Regressor

Estimators: 300 Trees

Max Depth: 30

Preprocessing: Standard Scaler

Validation: 80/20 Train-Test split
