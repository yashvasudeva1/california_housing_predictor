"""
California Housing Price Prediction using Random Forest Regressor
Following ML best practices including data preprocessing, model training, 
evaluation, and persistence.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_explore_data(filepath):
    """Load the dataset and perform basic exploration"""
    print("=" * 60)
    print("LOADING AND EXPLORING DATA")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
    return df

def preprocess_data(df):
    """Preprocess the data - handle outliers, split features and target"""
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature names: {list(X.columns)}")
    
    # Check for outliers using IQR method (optional - keeping all data for now)
    print(f"\nTarget variable range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target variable mean: {y.mean():.2f}")
    print(f"Target variable median: {y.median():.2f}")
    
    return X, y

def split_data(X, y, test_size=0.2):
    """Split data into training and testing sets"""
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Train/Test split ratio: {(1-test_size)*100:.0f}/{test_size*100:.0f}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print("\n" + "=" * 60)
    print("SCALING FEATURES")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeatures scaled using StandardScaler (mean=0, std=1)")
    print(f"Training set mean: {X_train_scaled.mean():.6f}")
    print(f"Training set std: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_random_forest(X_train, y_train):
    """Train Random Forest Regressor with optimized hyperparameters"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST REGRESSOR")
    print("=" * 60)
    
    # Initial model with good default parameters
    print("\n1. Training baseline model...")
    # -------------------------------
    # 1. Baseline Random Forest (no pre-fit)
    # -------------------------------
    rf_baseline = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\n2. Performing 5-fold cross-validation on baseline...")
    cv_scores = cross_val_score(
        rf_baseline,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print(f"   CV R² scores: {cv_scores}")
    print(f"   CV R² mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # -------------------------------
    # 2. Hyperparameter tuning (controlled search)
    # -------------------------------
    print("\n3. Performing hyperparameter tuning with GridSearchCV...")

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [15, 20, 25],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [3, 5],
        "max_features": [0.6, 0.8]
    }

    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    best_rf = grid_search.best_estimator_

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    
    return best_rf, grid_search.best_params_

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate the model and display metrics"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Testing metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTRAINING SET METRICS:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTESTING SET METRICS:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Check for overfitting
    overfit_indicator = train_r2 - test_r2
    print(f"\nOverfitting indicator (Train R² - Test R²): {overfit_indicator:.4f}")
    if overfit_indicator > 0.1:
        print("  ⚠ Warning: Model may be overfitting")
    else:
        print("  ✓ Model generalization looks good")
    
    # Feature importance
    print("\nFEATURE IMPORTANCE:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    return {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

def save_model(model, scaler, filepath='random_forest_model.pkl'):
    """Save the trained model and scaler to a pickle file"""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'model_type': 'RandomForestRegressor',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print(f"\n✓ Model saved successfully to: {filepath}")
    print(f"  File size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    return filepath

def load_model(filepath='random_forest_model.pkl'):
    """Load a saved model from pickle file"""
    with open(filepath, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    print(f"\n✓ Model loaded successfully from: {filepath}")
    print(f"  Model type: {model_artifacts['model_type']}")
    print(f"  Training date: {model_artifacts['training_date']}")
    
    return model_artifacts['model'], model_artifacts['scaler']

def main():
    """Main function to run the entire pipeline"""
    import os
    
    print("\n" + "=" * 60)
    print("CALIFORNIA HOUSING PRICE PREDICTION")
    print("RANDOM FOREST REGRESSOR IMPLEMENTATION")
    print("=" * 60)
    
    # 1. Load and explore data
    df = load_and_explore_data('California_Housing.csv')
    
    # 2. Preprocess data
    X, y = preprocess_data(df)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # 4. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 5. Train model
    model, best_params = train_random_forest(X_train_scaled, y_train)
    
    # 6. Evaluate model
    metrics = evaluate_model(
        model, 
        X_train_scaled, X_test_scaled, 
        y_train, y_test,
        feature_names=X.columns.tolist()
    )
    
    # 7. Save model
    model_path = save_model(model, scaler, 'random_forest_model.pkl')
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFinal Test R² Score: {metrics['test_r2']:.4f}")
    print(f"Final Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Model saved to: {model_path}")
    
    # Example of making predictions with the saved model
    print("\n" + "=" * 60)
    print("EXAMPLE: LOADING AND USING THE SAVED MODEL")
    print("=" * 60)
    
    loaded_model, loaded_scaler = load_model('random_forest_model.pkl')
    
    # Make a sample prediction
    sample = X_test.iloc[0:1]
    sample_scaled = loaded_scaler.transform(sample)
    prediction = loaded_model.predict(sample_scaled)
    actual = y_test.iloc[0]
    
    print(f"\nSample prediction:")
    print(f"  Predicted price: ${prediction[0]:.3f} (hundred thousands)")
    print(f"  Actual price:    ${actual:.3f} (hundred thousands)")
    print(f"  Difference:      ${abs(prediction[0] - actual):.3f}")

if __name__ == "__main__":
    main()
