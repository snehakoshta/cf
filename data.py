
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load Crop Recommendation Data
crop_data = pd.read_csv("E:/crop1/Crop_recommendation.csv")
print("Columns in Crop Dataset:", crop_data.columns)

# Load Fertilizer Recommendation Data
fertilizer_data = pd.read_csv("E:/crop1/Fertilizer Prediction.csv")
print("Original Columns in Fertilizer Dataset:", fertilizer_data.columns)

# Clean and rename Fertilizer Dataset columns
fertilizer_data.rename(columns={
    'Temparature': 'Temperature', 
    'Humidity ': 'Humidity', 
    'Moisture': 'Moisture', 
    'Soil Type': 'Soil Type',
    'Crop Type': 'Crop Type', 
    'Nitrogen': 'Nitrogen', 
    'Potassium': 'Potassium', 
    'Phosphorous': 'Phosphorous', 
    'Fertilizer Name': 'Fertilizer Name'
}, inplace=True)

print("Cleaned Fertilizer Dataset Columns:", fertilizer_data.columns)

# Encode categorical columns in Fertilizer Data
soil_type_mapping = {'Sandy': 0, 'Clayey': 1, 'Loamy': 2, 'Silt': 3}
crop_type_mapping = {'Wheat': 0, 'Rice': 1, 'Maize': 2, 'Barley': 3}

fertilizer_data['Soil Type'] = fertilizer_data['Soil Type'].map(soil_type_mapping)
fertilizer_data['Crop Type'] = fertilizer_data['Crop Type'].map(crop_type_mapping)

# Define features and target for Fertilizer Recommendation
fertilizer_features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
fertilizer_target = 'Fertilizer Name'

X_fertilizer = fertilizer_data[fertilizer_features]
y_fertilizer = fertilizer_data[fertilizer_target]

# Split Fertilizer Data
X_train_fertilizer, X_test_fertilizer, y_train_fertilizer, y_test_fertilizer = train_test_split(
    X_fertilizer, y_fertilizer, test_size=0.2, random_state=42
)

# Train Decision Tree and Random Forest for Fertilizer Recommendation
dt_model_fertilizer = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model_fertilizer.fit(X_train_fertilizer, y_train_fertilizer)

rf_model_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_fertilizer.fit(X_train_fertilizer, y_train_fertilizer)

# Evaluate Fertilizer Models
dt_fertilizer_predictions = dt_model_fertilizer.predict(X_test_fertilizer)
rf_fertilizer_predictions = rf_model_fertilizer.predict(X_test_fertilizer)

# Define features and target for Crop Recommendation
crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
crop_target = 'label'

X_crop = crop_data[crop_features]
y_crop = crop_data[crop_target]

# Split Crop Data
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

# Train Decision Tree and Random Forest for Crop Recommendation
dt_model_crop = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model_crop.fit(X_train_crop, y_train_crop)

rf_model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_crop.fit(X_train_crop, y_train_crop)

# Evaluate Crop Models
dt_crop_predictions = dt_model_crop.predict(X_test_crop)
rf_crop_predictions = rf_model_crop.predict(X_test_crop)

# Create Accuracy Table
accuracy_data = {
    "Model": ["Fertilizer Recommendation (DT)", "Crop Recommendation (DT)", "Fertilizer Recommendation (RF)", 
             "Crop Recommendation (RF)"],
    "Algorithm": ["Decision Tree", "Decision Tree","Random Forest", "Random Forest"],
    "Accuracy Score": [accuracy_score(y_test_fertilizer, dt_fertilizer_predictions), 
                       accuracy_score(y_test_crop, dt_crop_predictions), 
                       accuracy_score(y_test_fertilizer, rf_fertilizer_predictions), 
                       accuracy_score(y_test_crop, rf_crop_predictions)]}

accuracy_df = pd.DataFrame(accuracy_data)

# Apply Styling
styled_accuracy_df = accuracy_df.style.set_properties(**{
    'background-color': '#f5f5f5',
    'color': 'black',
    'border': '1px solid black',
    'font-size': '14px'
}).set_caption("Crop & Fertilizer Recommendation Model Accuracy Table")

# Display the Accuracy Table
print("\nCrop & Fertilizer Recommendation Model Accuracy Table:")
print(accuracy_df.to_string(index=False))  # For standard print format

# Save the trained models
with open('decision_tree_model_fertilizer.pkl', 'wb') as file:
    pickle.dump(dt_model_fertilizer, file)
print("Decision Tree model for Fertilizer Recommendation saved as 'decision_tree_model_fertilizer.pkl'")

with open('random_forest_model_fertilizer.pkl', 'wb') as file:
    pickle.dump(rf_model_fertilizer, file)
print("Random Forest model for Fertilizer Recommendation saved as 'random_forest_model_fertilizer.pkl'")

with open('decision_tree_model_crop.pkl', 'wb') as file:
    pickle.dump(dt_model_crop, file)
print("Decision Tree model for Crop Recommendation saved as 'decision_tree_model_crop.pkl'")

with open('random_forest_model_crop.pkl', 'wb') as file:
    pickle.dump(rf_model_crop, file)
print("Random Forest model for Crop Recommendation saved as 'random_forest_model_crop.pkl'")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert data types to float32 to save memory
X_train_fertilizer = X_train_fertilizer.astype(np.float32)
X_test_fertilizer = X_test_fertilizer.astype(np.float32)
X_train_crop = X_train_crop.astype(np.float32)
X_test_crop = X_test_crop.astype(np.float32)

# Define base models for Fertilizer Recommendation
fertilizer_estimators = [
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=12)),  # Reduced depth
    ('rf', RandomForestClassifier(n_estimators=10, random_state=12))  # Reduced estimators
]

# Define base models for Crop Recommendation
crop_estimators = [
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=12)),
    ('rf', RandomForestClassifier(n_estimators=10, random_state=12))
]

# Create Stacking Classifier for Fertilizer Recommendation
stack_model_fertilizer = StackingClassifier(
    estimators=fertilizer_estimators, final_estimator=LogisticRegression(), verbose=0
)
stack_model_fertilizer.fit(X_train_fertilizer, y_train_fertilizer)

# Create Stacking Classifier for Crop Recommendation
stack_model_crop = StackingClassifier(
    estimators=crop_estimators, final_estimator=LogisticRegression(), verbose=0
)
stack_model_crop.fit(X_train_crop, y_train_crop)

# Evaluate Stacking Models
stack_fertilizer_predictions = stack_model_fertilizer.predict(X_test_fertilizer)
stack_crop_predictions = stack_model_crop.predict(X_test_crop)

# Create Accuracy Table
accuracy_data = {
    "Model": [
        "Fertilizer Recommendation (DT)", "Crop Recommendation (DT)", "Fertilizer Recommendation (RF)", 
        "Crop Recommendation (RF)", "Fertilizer Recommendation (Stacking)", "Crop Recommendation (Stacking)"
    ],
    "Algorithm": [
        "Decision Tree", "Decision Tree", "Random Forest", "Random Forest",
        "Stacking (Decision Tree + Random Forest)", "Stacking (Decision Tree + Random Forest)"
    ],
    "Accuracy Score": [
        accuracy_score(y_test_fertilizer, dt_fertilizer_predictions),
        accuracy_score(y_test_crop, dt_crop_predictions),
        accuracy_score(y_test_fertilizer, rf_fertilizer_predictions),
        accuracy_score(y_test_crop, rf_crop_predictions),
        accuracy_score(y_test_fertilizer, stack_fertilizer_predictions),
        accuracy_score(y_test_crop, stack_crop_predictions)
    ]
}

accuracy_df = pd.DataFrame(accuracy_data)

# Apply Styling
styled_accuracy_df = accuracy_df.style.set_properties(**{
    'background-color': '#f5f5f5',
    'color': 'black',
    'border': '1px solid black',
    'font-size': '14px'
}).set_caption("Crop & Fertilizer Recommendation Model Accuracy Table")

# Display the Accuracy Table
print("\nCrop & Fertilizer Recommendation Model Accuracy Table:")
print(accuracy_df.to_string(index=False))  # For standard print format

# Save Stacking Models Using Joblib
joblib.dump(stack_model_fertilizer, 'stacking_model_fertilizer.pkl')
print("Stacking model for Fertilizer Recommendation saved as 'stacking_model_fertilizer.pkl'")

joblib.dump(stack_model_crop, 'stacking_model_crop.pkl')
print("Stacking model for Crop Recommendation saved as 'stacking_model_crop.pkl'")