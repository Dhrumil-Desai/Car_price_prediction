#TO RUN THE PROJECT TYPE THIS CODE INTO THE TERMINAL 
# python -m streamlit run app.py

import os
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl','rb'))

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

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

cardata = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cardata['name']=cardata['name'].apply(get_brand_name)

name = st.selectbox("Select Car Brand", cardata['name'].unique())
# Show selected car image
image_path = f"images/{name.lower()}.jpg"

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
    "Peugeot":"images/Peugeot.jpg"
}

if name in img_dict and os.path.exists(img_dict[name]):
    st.image(img_dict[name], width=350)

year = st.slider("Car Manufactured Year", 1994,2024)
km_driven = st.slider("No of kms drivern", 11,200000)
fuel = st.selectbox("Fuel Type", cardata['fuel'].unique())
seller_type = st.selectbox("Seller Type", cardata['seller_type'].unique())
transmission = st.selectbox("Transmission Type", cardata['transmission'].unique())
owner = st.selectbox("Owner Type", cardata['owner'].unique())
mileage = st.slider("Car Milage", 10,40)
engine = st.slider("Engine CC", 700, 5000)
max_power = st.slider("Max Power", 0,200)
seats = st.slider("No of Seats", 5,10)

if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    input_data_model['owner']=input_data_model['owner'].replace([         'First Owner',         'Second Owner',          'Third Owner',
    'Fourth & Above Owner',       'Test Drive Car'],[1,2,3,4,5]).astype(int)
    input_data_model['fuel']=input_data_model['fuel'].replace({'Diesel':1 ,'Petrol':2 ,'LPG': 3, 'CNG':4}).astype(int)
    input_data_model['seller_type'] = input_data_model['seller_type'].replace({'Individual': 1,'Dealer': 2,'Trustmark Dealer': 3}).astype(int)
    input_data_model['transmission'] = input_data_model['transmission'].replace({'Manual': 1,'Automatic': 2}).astype(int)
    input_data_model['name']=input_data_model['name'].replace({       'Maruti',         'Skoda',         'Honda',       'Hyundai',
        'Toyota',          'Ford',       'Renault',      'Mahindra',
          'Tata',     'Chevrolet',        'Datsun',          'Jeep',
 'Mercedes-Benz',    'Mitsubishi',          'Audi',    'Volkswagen',
           'BMW',        'Nissan',         'Lexus',        'Jaguar',
          'Land',            'MG',         'Volvo',        'Daewoo',
           'Kia',          'Fiat',         'Force',    'Ambassador',
         'Ashok',         'Isuzu',          'Opel'                 },
         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]).astype(int)
    car_price = model.predict(input_data_model)

    st.success(f"Estimated Car Price: ₹ {car_price[0]:,.2f}")
    if car_price[0] < 300000:
        st.info("💙 Budget Car")
    elif car_price[0] < 700000:
        st.success("💚 Mid Range Car")
    else:
        st.warning("💛 Premium Car")

    # st.divider()


    #st.balloons()