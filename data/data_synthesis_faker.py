from faker import Faker
import numpy as np
import pandas as pd

# Create synthetic dataset via faker
fake = Faker()

def generate_fake_delivery_data(n=1000):
    data = []
    for _ in range(n):
        record = {
            "Delivery_person_ID": fake.uuid4(),
            "Delivery_person_Age": np.random.randint(18, 50),
            "Delivery_person_Ratings": np.round(np.random.uniform(3.5, 5.0), 1),
            "Restaurant_latitude": np.random.uniform(10, 30),
            "Restaurant_longitude": np.random.uniform(70, 90),
            "Delivery_location_latitude": np.random.uniform(10, 30),
            "Delivery_location_longitude": np.random.uniform(70, 90),
            "Order_Date": fake.date_between(start_date="-1y", end_date="today").strftime("%d-%m-%Y"),
            "Time_Orderd": fake.time(),
            "Weatherconditions": np.random.choice(["Sunny", "Cloudy", "Fog", "Stormy"]),
            "Road_traffic_density": np.random.choice(["Low", "Medium", "High", "Jam"]),
            "Type_of_order": np.random.choice(["Snack", "Meal", "Drinks", "Buffet"]),
            "Type_of_vehicle": np.random.choice(["motorcycle", "scooter", "electric_scooter"]),
            "Festival": np.random.choice(["Yes", "No"]),
            "City": np.random.choice(["Urban", "Metropolitian", "Semi-Urban"]),
        }
        data.append(record)
    return pd.DataFrame(data)

synthetic_data = generate_fake_delivery_data(1000)
synthetic_data.to_csv("unassigned_orders_faker.csv", index=False)
print("Fake data generated!")