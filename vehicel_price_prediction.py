import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# === Load dataset ===
df = pd.read_csv("dataset.csv")

# Drop rows with missing target
df.dropna(subset=['price'], inplace=True)

# Drop unnecessary columns
df.drop(['name', 'description'], axis=1, inplace=True)

# Define features and target
X = df.drop("price", axis=1)
y = df["price"]

# Identify numerical and categorical features
numerical = ['year', 'mileage', 'cylinders', 'doors']
categorical = ['make', 'model', 'fuel', 'transmission', 'trim',
               'body', 'exterior_color', 'interior_color', 'drivetrain']

# Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ]
)

# Fit-transform the features
X_processed = preprocessor.fit_transform(X)

# === Save preprocessor ===
joblib.dump(preprocessor, "preprocessor.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Convert to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(np.array(y_train), dtype=tf.float32)
y_test = tf.convert_to_tensor(np.array(y_test), dtype=tf.float32)

# === Define the TensorFlow model ===
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),  # ✅ Explicit loss object
    metrics=['mae']
)

# === Train the model ===
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# === Evaluate the model ===
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.2f}")

# === Save the trained model ===
model.save("vehicle_price_model.h5")
