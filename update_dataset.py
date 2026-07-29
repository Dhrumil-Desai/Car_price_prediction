import os
import pandas as pd
import numpy as np

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "extended_car_details_1995_2026.csv")

def add_new_car_models():
    # 1. ચેક કરો કે ડેટાસેટ ફાઈલ અસ્તિત્વમાં છે કે નહીં
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} ફાઈલ મળી નથી! કૃપા કરીને સાચો પાથ ચકાસો.")
        return

    # ડેટાસેટ લોડ કરો
    df = pd.read_csv(CSV_PATH)
    print(f"📊 જૂના ડેટાસેટમાં કુલ Rows: {len(df)}")

    # 2. નવા Car Models નું લિસ્ટ (અહીં તમે તમારી નવી કારો સરળતાથી ઉમેરી શકો છો)
    new_cars_list = [
        # --- MARUTI ---
        {"brand": "Maruti", "name": "Maruti Fronx", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": 20.0, "engine": 1197, "max_power": 88.5, "seats": 5},
        {"brand": "Maruti", "name": "Maruti Brezza", "fuel": "CNG", "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner", "mileage": 25.5, "engine": 1462, "max_power": 86.6, "seats": 5},
        {"brand": "Maruti", "name": "Maruti Grand Vitara", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner", "mileage": 27.9, "engine": 1490, "max_power": 101.6, "seats": 5},

        # --- HYUNDAI ---
        {"brand": "Hyundai", "name": "Hyundai Exter", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": 19.4, "engine": 1197, "max_power": 81.8, "seats": 5},
        {"brand": "Hyundai", "name": "Hyundai Creta Facelift", "fuel": "Diesel", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner", "mileage": 18.0, "engine": 1493, "max_power": 114.4, "seats": 5},
        {"brand": "Hyundai", "name": "Hyundai Verna", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": 18.6, "engine": 1482, "max_power": 157.5, "seats": 5},

        # --- TATA ---
        {"brand": "Tata", "name": "Tata Punch", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": 20.0, "engine": 1199, "max_power": 86.6, "seats": 5},
        {"brand": "Tata", "name": "Tata Nexon Facelift", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner", "mileage": 17.0, "engine": 1199, "max_power": 118.3, "seats": 5},
        {"brand": "Tata", "name": "Tata Harrier", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner", "mileage": 16.8, "engine": 1956, "max_power": 167.6, "seats": 5},

        # --- MAHINDRA ---
        {"brand": "Mahindra", "name": "Mahindra Thar Roxx", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner", "mileage": 15.2, "engine": 2184, "max_power": 172.5, "seats": 5},
        {"brand": "Mahindra", "name": "Mahindra XUV700", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner", "mileage": 13.0, "engine": 1997, "max_power": 197.1, "seats": 7},
        {"brand": "Mahindra", "name": "Mahindra Scorpio N", "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": 14.5, "engine": 2184, "max_power": 172.5, "seats": 7},

        # --- HONDA ---
        {"brand": "Honda", "name": "Honda Elevate", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Automatic", "owner": "First Owner", "mileage": 16.9, "engine": 1498, "max_power": 119.3, "seats": 5},
        {"brand": "Honda", "name": "Honda City Hybrid", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner", "mileage": 27.1, "engine": 1498, "max_power": 124.3, "seats": 5},

        # --- TOYOTA ---
        {"brand": "Toyota", "name": "Toyota Urban Cruiser Taisor", "fuel": "Petrol", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner", "mileage": 20.0, "engine": 1197, "max_power": 88.5, "seats": 5},
        {"brand": "Toyota", "name": "Toyota Innova Hycross", "fuel": "Petrol", "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner", "mileage": 23.2, "engine": 1987, "max_power": 183.7, "seats": 7}
    ]

    # 3. ૧૯૯૫ થી ૨૦૨૬ સુધી દરેક વર્ષ માટે આ કારોનું વેલ્યુએશન બનાવવું (Auto-Generating Historical Rows)
    generated_rows = []
    
    for car in new_cars_list:
        # ૨૦૧૫ થી ૨૦૨૬ સુધીના મોડેલ્સ ઉમેરો
        for yr in range(2015, 2027):
            # વર્ષ પ્રમાણે કિલોમીટર ગણો (જૂની કાર = વધારે KM)
            kms = max(5000, (2026 - yr) * 12000)
            
            # કારની અંદાજિત ઓરિજિનલ શોરૂમ કિંમત પરથી સેલિંગ પ્રાઇઝ ગણો
            base_price = 800000 + (car['engine'] * 200)
            depreciated_price = max(40000.0, base_price * ((1 - 0.08) ** (2026 - yr)) - (kms * 1.1))

            new_row = {
                'brand': car['brand'],
                'model': car['name'],
                'name': car['name'],
                'year': int(yr),
                'selling_price': round(depreciated_price, 2),
                'km_driven': int(kms),
                'fuel': car['fuel'],
                'seller_type': car['seller_type'],
                'transmission': car['transmission'],
                'owner': car['owner'],
                'mileage': car['mileage'],
                'engine': car['engine'],
                'max_power': car['max_power'],
                'seats': car['seats']
            }
            generated_rows.append(new_row)

    # 4. Data Frame સાથે નવો ડેટા Merge કરો
    new_df = pd.DataFrame(generated_rows)

    # Column match કરવા માટે Missing Columns ની ચકાસણી
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    # જૂના અને નવા ડેટાને જોડો (Concat)
    updated_df = pd.concat([df, new_df], ignore_index=True)

    # ડુપ્લિકેટ રિમૂવ કરો (જો સરખી ડેટા રો હોય તો)
    updated_df.drop_duplicates(subset=['name', 'year', 'km_driven', 'fuel'], keep='first', inplace=True)

    # 5. CSV ફાઈલ પાછી ફ્રેશ સેવ કરો
    updated_df.to_csv(CSV_PATH, index=False)
    print(f"✅ સફળતાપૂર્વક નવો ડેટા ઉમેરાઈ ગયો!")
    print(f"📈 હવે ડેટાસેટમાં કુલ Rows: {len(updated_df)}")

if __name__ == "__main__":
    add_new_car_models()