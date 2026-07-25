import os
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the ML model
model = pk.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

# Content of Sidebar 
st.sidebar.title("About Project")
st.sidebar.info(
    "This Machine Learning project predicts the selling price of a used car based on various features."
)
st.sidebar.divider()
st.sidebar.markdown("### 🚘 Features Used")
st.sidebar.markdown("""
- 🚘 Car Brand
- 📅 Manufacturing Year
- 🛣️ Kilometers Driven
- ⛽ Fuel Type
- 🏪 Seller Type
- ⚙️ Transmission
- 👤 Owner Type
- 📈 Mileage
- 🔧 Engine (CC)
- 💨 Max Power
- 💺 Number of Seats
""")
st.sidebar.divider()
st.sidebar.success("👨‍💻 Developed by Dhrumil Desai")

st.header("Car Price Prediction ML Model")

# To Upload the Data
cardata = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cardata['name'] = cardata['name'].apply(get_brand_name)

# ---------------------------------------------------------
# To set the Default Values by using Session State 
# ---------------------------------------------------------
if 'brand' not in st.session_state:
    st.session_state.brand = cardata['name'].unique()[0]
if 'year' not in st.session_state:
    st.session_state.year = 1994
if 'km_driven' not in st.session_state:
    st.session_state.km_driven = 11
if 'fuel' not in st.session_state:
    st.session_state.fuel = cardata['fuel'].unique()[0]
if 'seller_type' not in st.session_state:
    st.session_state.seller_type = cardata['seller_type'].unique()[0]
if 'transmission' not in st.session_state:
    st.session_state.transmission = cardata['transmission'].unique()[0]
if 'owner' not in st.session_state:
    st.session_state.owner = cardata['owner'].unique()[0]
if 'mileage' not in st.session_state:
    st.session_state.mileage = 10
if 'engine' not in st.session_state:
    st.session_state.engine = 700
if 'max_power' not in st.session_state:
    st.session_state.max_power = 0
if 'seats' not in st.session_state:
    st.session_state.seats = 5

# To reset the default value, use RESET Function
def reset_fields():
    st.session_state.brand = cardata['name'].unique()[0]
    st.session_state.year = 1994
    st.session_state.km_driven = 11
    st.session_state.fuel = cardata['fuel'].unique()[0]
    st.session_state.seller_type = cardata['seller_type'].unique()[0]
    st.session_state.transmission = cardata['transmission'].unique()[0]
    st.session_state.owner = cardata['owner'].unique()[0]
    st.session_state.mileage = 10
    st.session_state.engine = 700
    st.session_state.max_power = 0
    st.session_state.seats = 5

# ---------------------------------------------------------
# Form Inputs with Key
# ---------------------------------------------------------
brands_list = cardata['name'].unique().tolist()
name = st.selectbox("Select Car Brand", brands_list, key='brand')

# To display the Image of all Brands
img_dict = {
    "Maruti": "images/Maruti.jpg", 
    "Honda": "images/Honda.jpg",
    "Hyundai": "images/Hyundai.jpg",
    "Toyota": "images/Toyota.jpg", 
    "Ford": "images/Ford.jpg", 
    "Renault": "images/Renault.jpg",
    "Mahindra": "images/Mahindra.jpg", 
    "Tata": "images/Tata.jpg", 
    "BMW": "images/BMW.jpg",
    "Audi": "images/Audi.jpg", 
    "Mercedes-Benz": "images/Mercedes-Benz.jpg", 
    "Jeep": "images/Jeep.jpg",
    "Volkswagen": "images/Volkswagen.jpg", 
    "Volvo": "images/Volvo.jpg", 
    "MG": "images/MG.jpg",
    "Nissan": "images/Nissan.jpg", 
    "Kia": "images/Kia.jpg", 
    "Jaguar": "images/Jaguar.jpg",
    "Lexus": "images/Lexus.jpg", 
    "Land": "images/Land.jpg", 
    "Skoda": "images/Skoda.jpg",
    "Chevrolet": "images/Chevrolet.jpg", 
    "Datsun": "images/Datsun.jpg", 
    "Daewoo": "images/Daewoo.jpg",
    "Fiat": "images/Fiat.jpg", 
    "Force": "images/Force.jpg", 
    "Isuzu": "images/Isuzu.jpg",
    "Mitsubishi": "images/Mitsubishi.jpg", 
    "Opel": "images/Opel.jpg", 
    "Ashok": "images/Ashok.jpg",
    "Ambassador": "images/Ambassador.jpg", 
    "Peugeot": "images/Peugeot.jpg"
}

if name in img_dict and os.path.exists(img_dict[name]):
    st.image(img_dict[name], width=350)

year = st.slider("Car Manufactured Year", 1994, 2024, key='year')
km_driven = st.slider("No of kms drivern", 11, 200000, key='km_driven')
fuel = st.selectbox("Fuel Type", cardata['fuel'].unique(), key='fuel')
seller_type = st.selectbox("Seller Type", cardata['seller_type'].unique(), key='seller_type')
transmission = st.selectbox("Transmission Type", cardata['transmission'].unique(), key='transmission')
owner = st.selectbox("Owner Type", cardata['owner'].unique(), key='owner')
mileage = st.slider("Car Milage", 10, 40, key='mileage')
engine = st.slider("Engine CC", 700, 5000, key='engine')
max_power = st.slider("Max Power", 0, 200, key='max_power')
seats = st.slider("No of Seats", 5, 10, key='seats')

# ---------------------------------------------------------
# Predict AND Reset Button
# ---------------------------------------------------------
col1, col2 = st.columns([1, 6])

with col1:
    predict_btn = st.button("🚗 Predict")

with col2:
    st.button("🔄 Reset", on_click=reset_fields)

if predict_btn:
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    input_data_model['owner'] = input_data_model['owner'].replace([
        'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'
    ], [1, 2, 3, 4, 5]).astype(int)

    input_data_model['fuel'] = input_data_model['fuel'].replace({'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}).astype(int)
    input_data_model['seller_type'] = input_data_model['seller_type'].replace({'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}).astype(int)
    input_data_model['transmission'] = input_data_model['transmission'].replace({'Manual': 1, 'Automatic': 2}).astype(int)
    
    # Dict mapping
    brand_map = {
        'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
        'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13,
        'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
        'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25, 'Fiat': 26,
        'Force': 27, 'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31
    }
    input_data_model['name'] = input_data_model['name'].map(brand_map).fillna(1).astype(int)

    car_price = model.predict(input_data_model)

    st.success(f"Estimated Car Price: ₹ {car_price[0]:,.2f}")
    if car_price[0] < 300000:
        st.info("💙 Budget Car")
    elif car_price[0] < 700000:
        st.success("💚 Mid Range Car")
    else:
        st.warning("💛 Premium Car")