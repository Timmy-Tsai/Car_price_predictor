import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor

# Create an app instance
app = Flask(__name__)

# Correct loading of pickle files
with open('models/Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/Brand_Encoder.pkl', 'rb') as f:
    Brand_Encoder = pickle.load(f)

with open('models/Model_Encoder.pkl', 'rb') as f:
    Model_Encoder = pickle.load(f)

with open('models/OneHot_Encoder.pkl', 'rb') as f:
    OneHot_Encoder = pickle.load(f)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        Brand = request.form.get('Brand')
        Model = request.form.get('Model')
        Fuel = request.form.get('Fuel')
        Transmission = request.form.get('Transmission')
        Year = int(request.form.get('Year'))
        EngineSize = float(request.form.get('EngineSize'))
        Mileage = int(request.form.get('Mileage'))
        Doors = int(request.form.get('Doors'))
        OwnerCount = int(request.form.get('OwnerCount'))

        # Create a DataFrame
        input_df = pd.DataFrame({
            'Brand': [Brand],
            'Model': [Model],
            'Year': [Year],
            'EngineSize': [EngineSize],
            'Fuel': [Fuel],
            'Transmission': [Transmission],
            'Mileage': [Mileage],
            'Doors': [Doors],
            'OwnerCount': [OwnerCount]
        })

        # Encode Brand and Model using mean target encoding
        input_df['Encoded_Brand'] = input_df['Brand'].map(Brand_Encoder)
        input_df['Encoded_Model'] = input_df['Model'].map(Model_Encoder)
        input_df['Encoded_Brand'].fillna(Brand_Encoder.mean(), inplace=True)
        input_df['Encoded_Model'].fillna(Model_Encoder.mean(), inplace=True)
        input_df.drop(['Brand', 'Model'], axis=1, inplace=True)

        # One-hot encode Fuel and Transmission
        categorical_cols = ['Fuel', 'Transmission']
        encoded_array = OneHot_Encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=OneHot_Encoder.get_feature_names_out(categorical_cols))

        # Merge encoded columns with input data
        input_df_encoded = input_df.drop(columns=categorical_cols).reset_index(drop=True)
        input_data = pd.concat([input_df_encoded, encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)

        if output < 0:
            return render_template('index.html', prediction_text="Sorry, you cannot sell this car")
        else:
            return render_template('index.html', prediction_text=f"Car is worth: ${output}")

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
