import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Sample dataset (you can replace this with CSV later)
data = {
    'study_hours': [2, 4, 6, 8, 10],
    'attendance': [60, 70, 80, 90, 95],
    'final_score': [50, 60, 70, 85, 95]
}

df = pd.DataFrame(data)

# Features and target
X = df[['study_hours', 'attendance']]
y = df['final_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)
