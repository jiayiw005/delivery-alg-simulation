import pandas as pd
import numpy as np
from geopy.distance import geodesic
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def load_and_preprocess_data(filepath_or_df):
    """Load and preprocess data with enhanced feature engineering"""
    if isinstance(filepath_or_df, str):
        df = pd.read_csv(filepath_or_df).reset_index(drop=True)
    else:
        df = filepath_or_df.copy()
    
    # Time features
    if 'Time_taken(min)' in df.columns:
        df['Time_taken(min)'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)
    if 'Time_Orderd' in df.columns:
        df['Order_Hour'] = pd.to_datetime(
            df['Time_Orderd'], 
            format='%H:%M', 
            errors='coerce'
        ).dt.hour.fillna(12)
        df['Is_Peak_Hour'] = df['Order_Hour'].isin([7,8,9,17,18,19]).astype(int)
    
    # Spatial features
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
        df['Direction'] = np.arctan2(
            df['Delivery_location_latitude'] - df['Restaurant_latitude'],
            df['Delivery_location_longitude'] - df['Restaurant_longitude']
        )
    
    # Temporal features
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['Order_Date'])
        df['Day_of_week'] = df['Order_Date'].dt.weekday
        df['Is_Weekend'] = df['Day_of_week'].isin([5,6]).astype(int)
        df['Month'] = df['Order_Date'].dt.month
    
    return df

def build_xgboost_pipeline():
    """Build optimized pipeline with XGBoost"""
    categorical_features = ['Weatherconditions', 'Road_traffic_density', 
                          'Type_of_vehicle', 'City', 'Festival']
    numeric_features = ['Distance_km', 'Order_Hour', 'Is_Peak_Hour',
                       'Direction', 'Day_of_week', 'Month']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        ))
    ])

def train_and_evaluate(df):
    """Train and evaluate the model"""
    features = ['Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle',
               'City', 'Festival', 'Distance_km', 'Order_Hour', 'Is_Peak_Hour',
               'Direction', 'Day_of_week', 'Month']
    
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df['Time_taken(min)'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model = build_xgboost_pipeline()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: {mae:.2f} minutes")
    
    return model, features

if __name__ == "__main__":
    try:
        df = load_and_preprocess_data("train.csv")
        
        model, features = train_and_evaluate(df)
        
        import joblib
        joblib.dump(model, 'eta_predictor_xgboost.pkl')
        print("Model saved as 'eta_predictor_xgboost.pkl'")
        
    except Exception as e:
        print(f"Error: {str(e)}")