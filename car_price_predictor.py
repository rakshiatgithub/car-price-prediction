import pandas as pd
import numpy as np
import streamlit as st
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('car_data.csv')

# Cleaning the data
df['Price'] = df['Price'].str.replace(',', '').str.replace('Ask For Price', '0')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '')
df = df.dropna(subset=['kms_driven', 'Price'])
df = df[df['kms_driven'].str.isdigit()]
df['kms_driven'] = df['kms_driven'].astype(int)
df = df[df['Price'] != 0]

# Encode categorical variables
le_name = LabelEncoder()
le_company = LabelEncoder()
le_fuel = LabelEncoder()

df['name'] = le_name.fit_transform(df['name'])
df['company'] = le_company.fit_transform(df['company'])
df['fuel_type'] = le_fuel.fit_transform(df['fuel_type'])

# Prepare data
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Train model
# model = LinearRegression()
model = RandomForestRegressor()

model.fit(X, y)

# Streamlit UI
# st.title('🚗 Car Price Predictor')
st.write("#### Available data ranges:")
st.write(f"- Years: {df['year'].min()} to {df['year'].max()}")
st.write(f"- Kms Driven: {df['kms_driven'].min():,} to {df['kms_driven'].max():,}")

# Dropdown options
car_names = le_name.classes_
company_names = le_company.classes_
fuel_type_names = le_fuel.classes_

# User inputs
selected_name = st.selectbox('Car Model Name', car_names)
selected_company = st.selectbox('Company', company_names)
year = st.number_input('Year', min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input('Kms Driven', min_value=0)
selected_fuel_type = st.selectbox('Fuel Type', fuel_type_names)

# Predict button
if st.button('Predict Price'):
    # Encode input values
    name_encoded = le_name.transform([selected_name])[0]
    company_encoded = le_company.transform([selected_company])[0]
    fuel_type_encoded = le_fuel.transform([selected_fuel_type])[0]

    # Prediction
    input_data = np.array([[name_encoded, company_encoded, year, kms_driven, fuel_type_encoded]])
    st.write("Debug - Encoded input sent to model:", input_data)

    prediction = model.predict(input_data)[0]
    final_price = max(0, int(prediction))

    if final_price == 0:
        st.warning("⚠️ The predicted price is ₹0. Please try different input values.")
    else:
        st.success(f"✅ Predicted Car Price: ₹ {final_price:,}")
