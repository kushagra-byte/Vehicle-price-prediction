import pandas as pd
import tensorflow as tf
import joblib

# === Load model and preprocessor ===
model = tf.keras.models.load_model("vehicle_price_model.h5")
preprocessor = joblib.load("preprocessor.pkl")

# === Sample vehicle input ===
sample_vehicle = {
    'year': 2020,
    'mileage': 25000,
    'cylinders': 4,
    'doors': 4,
    'make': 'Toyota',
    'model': 'Camry',
    'fuel': 'Gasoline',
    'transmission': 'Automatic',
    'trim': 'LE',
    'body': 'Sedan',
    'exterior_color': 'White',
    'interior_color': 'Black',
    'drivetrain': 'Front-wheel Drive'
}

# === Convert to DataFrame ===
input_df = pd.DataFrame([sample_vehicle])

# === Preprocess ===
processed = preprocessor.transform(input_df)
if hasattr(processed, "toarray"):
    processed = processed.toarray()

# === Convert to Tensor ===
tensor_input = tf.convert_to_tensor(processed, dtype=tf.float32)

# === Predict ===
predicted_price = model.predict(tensor_input)
print(f"Predicted Vehicle Price: ${predicted_price[0][0]:,.2f}")
