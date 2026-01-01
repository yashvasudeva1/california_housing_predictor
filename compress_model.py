import pickle
import joblib
import os

# 1. Load the heavy pickle file
print("Loading original model (this might take a few seconds)...")
with open('random_forest_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# 2. Save it as a compressed joblib file
# 'compress=3' offers a great balance of speed vs size
print("Compressing and saving as 'random_forest_model.joblib'...")
joblib.dump(model_data, 'random_forest_model.joblib', compress=3)

# 3. Check the new size
original_size = os.path.getsize('random_forest_model.pkl') / (1024 * 1024)
new_size = os.path.getsize('random_forest_model.joblib') / (1024 * 1024)

print(f"\nDone!")
print(f"Original Size: {original_size:.2f} MB")
print(f"New Size:      {new_size:.2f} MB")