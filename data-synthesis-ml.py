import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from faker import Faker
from datetime import datetime, timedelta

def generate_unassigned_orders_ml(historical_data, num_orders=1000):
    """
    Generate synthetic unassigned orders using ML techniques
    
    Parameters:
    historical_data (DataFrame): Past order data for pattern learning
    num_orders (int): Number of unassigned orders to generate
    
    Returns:
    DataFrame: Synthetic unassigned orders with realistic patterns
    """
    fake = Faker()
    np.random.seed(42)
    
    # 1. Clean and prepare historical data
    df = historical_data.copy()
    if 'Time_taken(min)' in df.columns:
        df['Time_taken(min)'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)
    loc_cols = ['Restaurant_latitude', 'Restaurant_longitude',
               'Delivery_location_latitude', 'Delivery_location_longitude']
    df = df.dropna(subset=loc_cols)
    
    # 2. Spatial Pattern Learning with GMM
    print("Learning spatial patterns with Gaussian Mixture Models...")
    locations = df[loc_cols]
    scaler = StandardScaler()
    scaled_locs = scaler.fit_transform(locations)
    
    # Train GMM (adjust n_components as needed)
    n_components = min(5, len(df)//10)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(scaled_locs)
    
    synthetic_locs, _ = gmm.sample(num_orders)
    synthetic_locs = scaler.inverse_transform(synthetic_locs)
    
    # 3. Handle categorical variables robustly
    print("Processing categorical variables...")
    cat_cols = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 
               'Type_of_vehicle', 'City', 'Festival']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    synthetic_cats = pd.DataFrame()
    for col in cat_cols:
        value_counts = df[col].value_counts(normalize=True)
        
        if len(value_counts) == 0:
            synthetic_cats[col] = ['Unknown'] * num_orders
            continue
            
        # Normalize probabilities to ensure they sum to 1
        probs = value_counts.values
        probs = probs / probs.sum()
        
        probs[-1] = 1 - probs[:-1].sum()  
        
        try:
            synthetic_cats[col] = np.random.choice(
                value_counts.index,
                size=num_orders,
                p=probs
            )
        except ValueError as e:
            print(f"Error with column {col}: {str(e)}")
            print(f"Probabilities sum: {probs.sum()}")
            print(f"Probabilities: {probs}")
            # Fallback to uniform distribution
            synthetic_cats[col] = np.random.choice(
                value_counts.index,
                size=num_orders
            )
    
    # 4. Generate temporal patterns
    print("Generating temporal patterns...")
    order_times = []
    current_time = datetime.now()
    
    hour_probs = np.array([0.02] * 24)
    peak_hours = [11, 12, 13, 18, 19, 20]
    hour_probs[peak_hours] = 0.15
    hour_probs = hour_probs / hour_probs.sum()
    
    for _ in range(num_orders):
        hour = np.random.choice(range(24), p=hour_probs)
        minute = np.random.randint(0, 60)
        day_offset = np.random.randint(0, 3)
        order_time = current_time.replace(hour=hour, minute=minute) - timedelta(days=day_offset)
        order_times.append(order_time)
    
    # 5. Combine all features
    print("Combining features...")
    synthetic = pd.DataFrame()
    
    # Spatial features
    for i, col in enumerate(loc_cols):
        synthetic[col] = synthetic_locs[:, i]
    
    # Calculate haversine distance
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    synthetic['Distance_km'] = haversine(
        synthetic['Restaurant_latitude'], synthetic['Restaurant_longitude'],
        synthetic['Delivery_location_latitude'], synthetic['Delivery_location_longitude']
    )
    
    # Categorical features
    for col in cat_cols:
        synthetic[col] = synthetic_cats[col]
    
    # Temporal features
    synthetic['Order_DateTime'] = order_times
    synthetic['Order_Date'] = [dt.date() for dt in order_times]
    synthetic['Order_Time'] = [dt.time() for dt in order_times]
    synthetic['Day_of_Week'] = [dt.weekday() for dt in order_times]
    synthetic['Is_Weekend'] = synthetic['Day_of_Week'].isin([5, 6]).astype(int)
    
    # Order characteristics
    synthetic['Order_ID'] = [f'UNASSIGNED_{fake.unique.uuid4()[:8]}' for _ in range(num_orders)]
    
    # Order size based on type (with fallback)
    type_size_map = {
        'Snack': [0.7, 0.25, 0.05],
        'Meal': [0.2, 0.5, 0.3],
        'Drinks': [0.8, 0.15, 0.05],
        'Buffet': [0.1, 0.3, 0.6],
        'Unknown': [0.5, 0.3, 0.2]
    }
    
    synthetic['Order_Size'] = synthetic.apply(
        lambda row: np.random.choice(
            ['Small', 'Medium', 'Large'],
            p=type_size_map.get(str(row.get('Type_of_order', 'Unknown')), [0.5, 0.3, 0.2])
        ),
        axis=1
    )
    
    # Priority assignment
    synthetic['Priority'] = synthetic.apply(
        lambda row: 'Express' if (row['Order_DateTime'].hour in peak_hours 
                              and np.random.random() > 0.7) else
                   'High' if np.random.random() > 0.85 else
                   'Normal',
        axis=1
    )
    
    print("Successfully generated synthetic unassigned orders!")
    return synthetic

# Generate unassigned orders
if __name__ == "__main__":
    try:
        
        historical_orders = pd.read_csv("/data/train.csv")
        print(f"Loaded {len(historical_orders)} historical orders")
        
        unassigned_orders = generate_unassigned_orders_ml(historical_orders, num_orders=1000)
        
        # Save to csv
        output_path = "/data/unassigned_orders_ml.csv"
        unassigned_orders.to_csv(output_path, index=False)
        print(f"Saved {len(unassigned_orders)} unassigned orders to {output_path}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")