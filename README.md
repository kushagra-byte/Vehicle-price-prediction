# Vehicle-price-prediction

#\Vehicle Price Prediction using TensorFlow & Scikit-learn
This project builds and trains a deep learning regression model to predict the price of a vehicle based on its specifications. It uses a TensorFlow neural network for prediction and a scikit-learn pipeline for data preprocessing.

#Dataset Requirements
Place your dataset as dataset.csv with the following structure:

🧩 Features Used:
Numerical: year, mileage, cylinders, doors

Categorical: make, model, fuel, transmission, trim, body, exterior_color, interior_color, drivetrain

🎯 Target:
price (vehicle price, numeric)

🔻 Automatically Dropped:
name, description

# Setup Instructions
Install required packages:

pip install pandas numpy tensorflow scikit-learn joblib

#Model Training
Run the training script:

vehicle_price_prediction.py

Evaluation Output
The model is evaluated using:

Loss: Mean Squared Error (MSE)

Metric: Mean Absolute Error (MAE)

Example Output:

Test MAE: 1785.42

After training and saving the model 
run predict_price.py 

