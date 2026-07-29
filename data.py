import random
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
num_samples = 10000  # Change this to adjust the size of your dataset
years = list(range(1995, 2027))  # 1995 to 2026

# Sample car inventory templates (Brand, Model, Fuel Options, Engine, Power, Torque, Seats)
car_templates = [
    {
        "name": "Maruti Swift",
        "fuel_pool": ["Petrol", "Diesel"],
        "engine": "1197 CC",
        "power": "81.80 bhp",
        "torque": "113Nm@ 4200rpm",
        "seats": 5.0,
        "mileage": "21.21 kmpl",
    },
    {
        "name": "Hyundai Creta",
        "fuel_pool": ["Petrol", "Diesel"],
        "engine": "1497 CC",
        "power": "113.45 bhp",
        "torque": "250Nm@ 1500-2750rpm",
        "seats": 5.0,
        "mileage": "16.8 kmpl",
    },
    {
        "name": "Honda City",
        "fuel_pool": ["Petrol", "Diesel"],
        "engine": "1498 CC",
        "power": "119.35 bhp",
        "torque": "145Nm@ 4600rpm",
        "seats": 5.0,
        "mileage": "17.4 kmpl",
    },
    {
        "name": "Mahindra Scorpio",
        "fuel_pool": ["Diesel"],
        "engine": "2179 CC",
        "power": "140 bhp",
        "torque": "320Nm@ 1500-2800rpm",
        "seats": 7.0,
        "mileage": "15.0 kmpl",
    },
    {
        "name": "Toyota Fortuner",
        "fuel_pool": ["Diesel", "Petrol"],
        "engine": "2755 CC",
        "power": "174.5 bhp",
        "torque": "420Nm@ 1400-2600rpm",
        "seats": 7.0,
        "mileage": "10.0 kmpl",
    },
    {
        "name": "Tata Nexon",
        "fuel_pool": ["Petrol", "Diesel", "CNG"],
        "engine": "1199 CC",
        "power": "118.35 bhp",
        "torque": "170Nm@ 1750-4000rpm",
        "seats": 5.0,
        "mileage": "17.5 kmpl",
    },
    {
        "name": "Maruti Alto",
        "fuel_pool": ["Petrol", "CNG"],
        "engine": "796 CC",
        "power": "47.3 bhp",
        "torque": "69Nm@ 3500rpm",
        "seats": 4.0,
        "mileage": "22.05 kmpl",
    },
    {
        "name": "Ford EcoSport",
        "fuel_pool": ["Petrol", "Diesel"],
        "engine": "1498 CC",
        "power": "98.98 bhp",
        "torque": "215Nm@ 1750-2500rpm",
        "seats": 5.0,
        "mileage": "21.7 kmpl",
    },
]

seller_types = ["Individual", "Dealer", "Trustmark Dealer"]
transmissions = ["Manual", "Automatic"]
owners = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]

data = []

for _ in range(num_samples):
  template = random.choice(car_templates)
  year = np.random.choice(
      years,
      p=np.linspace(0.005, 0.08, len(years))
      / np.sum(np.linspace(0.005, 0.08, len(years))),
  )  # More weight to newer cars

  fuel = random.choice(template["fuel_pool"])
  seller_type = random.choice(seller_types)
  transmission = random.choice(transmissions)
  owner = random.choice(owners)

  # Scale kilometers driven based on car age
  car_age = 2026 - year
  km_driven = int(
      np.clip(
          np.random.normal(loc=car_age * 12000, scale=5000), 5000, 350000
      )
  )

  # Price logic: Base price depreciates with age and increases with features/transmission
  base_price = np.random.uniform(300000, 1800000)
  depreciation_rate = 0.06
  selling_price = int(
      base_price * ((1 - depreciation_rate) ** car_age)
      + np.random.normal(0, 25000)
  )
  selling_price = max(
      40000, selling_price
  )  # Floor price for oldest running cars

  # Adjust automatic transmission and diesel/petrol price adjustments
  if transmission == "Automatic":
    selling_price = int(selling_price * 1.15)

  data.append({
      "name": f"{template['name']} {fuel}",
      "year": int(year),
      "selling_price": int(selling_price),
      "km_driven": km_driven,
      "fuel": fuel,
      "seller_type": seller_type,
      "transmission": transmission,
      "owner": owner,
      "mileage": template["mileage"],
      "engine": template["engine"],
      "max_power": template["power"],
      "torque": template["torque"],
      "seats": template["seats"],
  })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("extended_car_details_1995_2026.csv", index=False)
print(
    "Dataset successfully generated! Shape:",
    df.shape,
    "| Saved as 'extended_car_details_1995_2026.csv'",
)
print(df.head())