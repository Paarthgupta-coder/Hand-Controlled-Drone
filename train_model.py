import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load your custom dataset
# We check if the file exists first to avoid errors
try:
    data = pd.read_csv('gesture_data.csv', header=None)
    print(f"Loaded {len(data)} rows of gesture data.")
except FileNotFoundError:
    print("Error: gesture_data.csv not found. Please run collect_data.py first.")
    exit()

X = data.iloc[:, :-1]  # The 42 landmark coordinates (X and Y for 21 points)
y = data.iloc[:, -1]   # The labels (UP, DOWN, FLIP, IDLE, etc.)

# 2. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and Train the Model
# Random Forest is highly effective for coordinate-based classification
print("Training the Custom AI Model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Verify Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model Training Complete!")
print(f"Validation Accuracy: {accuracy:.2f}%")

# 5. Save the 'Brain'
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Brain saved as 'gesture_model.pkl'. You can now run drone_ai_control.py!")