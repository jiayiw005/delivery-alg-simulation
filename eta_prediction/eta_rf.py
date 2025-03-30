import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def load_and_preprocess_data(filepath_or_df):
    """Load and preprocess the delivery data"""
    if isinstance(filepath_or_df, str):
        df = pd.read_csv(filepath_or_df).reset_index(drop=True)
    else:
        df = filepath_or_df.copy()
    
    if 'Time_taken(min)' in df.columns:
        df['Time_taken(min)'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)
    
    def parse_time(time_str):
        if pd.isna(time_str):
            return np.nan
        for fmt in ('%H:%M', '%H.%M', '%I:%M %p', '%I.%M %p'):
            try:
                return pd.to_datetime(time_str, format=fmt).time()
            except:
                continue
        return np.nan
    
    if 'Time_Orderd' in df.columns:
        df['Order_Time'] = df['Time_Orderd'].apply(parse_time)
        df['Order_Hour'] = df['Order_Time'].apply(lambda x: x.hour if pd.notna(x) else np.nan)
        if df['Order_Hour'].notna().any():
            mode_hour = df['Order_Hour'].mode()[0]
            df['Order_Hour'] = df['Order_Hour'].fillna(mode_hour)
        else:
            df['Order_Hour'] = 12
    
    # Distance calculation
    loc_cols = ['Restaurant_latitude', 'Restaurant_longitude',
               'Delivery_location_latitude', 'Delivery_location_longitude']
    if all(col in df.columns for col in loc_cols):
        df['Distance_km'] = df.apply(
            lambda row: geodesic(
                (row['Restaurant_latitude'], row['Restaurant_longitude']),
                (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
            ).km,
            axis=1
        )
    
    # Temporal features
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['Order_Date'])
        df['Is_Weekend'] = df['Order_Date'].dt.weekday.isin([5, 6]).astype(int)
        df['Month'] = df['Order_Date'].dt.month
        df['Day_of_week'] = df['Order_Date'].dt.weekday
    
    if 'Order_Hour' in df.columns:
        df['Is_Rush_Hour'] = df['Order_Hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
    
    return df

def build_model_pipeline():
    """Build the ML pipeline for ETA prediction"""
    categorical_features = ['Weatherconditions', 'Road_traffic_density', 
                          'Type_of_vehicle', 'City', 'Festival']
    numeric_features = ['Distance_km', 'Order_Hour', 'Is_Weekend', 
                       'Month', 'Day_of_week', 'Is_Rush_Hour']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ])
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        ))
    ])

def train_and_evaluate(df):
    """Train and evaluate the model"""
    # Feature definitions
    possible_features = ['Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle', 
                        'City', 'Festival', 'Distance_km', 'Order_Hour', 'Is_Weekend',
                        'Month', 'Day_of_week', 'Is_Rush_Hour']
    features = [f for f in possible_features if f in df.columns]
    
    X = df[features]
    y = df['Time_taken(min)']
    
    valid_rows = y.notna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model = build_model_pipeline()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: {mae:.2f} minutes")
    
    return model, features

if __name__ == "__main__":
    try:
        df = load_and_preprocess_data("train.csv")
        
        if len(df) < 100:
            raise ValueError("Insufficient data after preprocessing")
            
        model, features_used = train_and_evaluate(df)
        
    except Exception as e:
        print(f"Error: {str(e)}")