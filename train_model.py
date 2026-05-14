# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import joblib
import os

print("🌾 Training Crop Prediction Models...")
print("="*50)

# Create data directory
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Generate comprehensive dataset
np.random.seed(42)

# 1. Yield Dataset (50,000 samples)
print("\n📊 Generating yield dataset...")

crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Potato', 
         'Tomato', 'Apple', 'Banana', 'Grapes', 'Soybean', 'Groundnut',
         'Barley', 'Sunflower', 'Chickpea', 'Pigeonpeas', 'Lentil',
         'Mango', 'Orange', 'Papaya', 'Coconut', 'Coffee', 'Tea']

data_yield = []
for crop in crops:
    for _ in range(2000):
        # Crop-specific parameters
        if crop == 'Rice':
            temp, rain = np.random.uniform(22, 32), np.random.uniform(1000, 2500)
            yield_base = 4500
            N, P, K = np.random.uniform(80, 120), np.random.uniform(35, 55), np.random.uniform(40, 65)
        elif crop == 'Wheat':
            temp, rain = np.random.uniform(15, 25), np.random.uniform(300, 600)
            yield_base = 3500
            N, P, K = np.random.uniform(70, 110), np.random.uniform(35, 60), np.random.uniform(40, 70)
        elif crop == 'Apple':
            temp, rain = np.random.uniform(10, 25), np.random.uniform(800, 1200)
            yield_base = 25000
            N, P, K = np.random.uniform(80, 120), np.random.uniform(40, 70), np.random.uniform(50, 90)
        else:
            temp, rain = np.random.uniform(18, 30), np.random.uniform(500, 1500)
            yield_base = np.random.uniform(2000, 5000)
            N, P, K = np.random.uniform(60, 120), np.random.uniform(30, 60), np.random.uniform(40, 80)
        
        # Generate other parameters
        soil_moisture = np.random.uniform(30, 80)
        soil_ph = np.random.uniform(5.5, 7.5)
        air_temp = temp + np.random.uniform(-2, 2)
        air_moisture = np.random.uniform(50, 90)
        soil_temp = temp + np.random.uniform(-3, 3)
        sunlight = np.random.uniform(6, 10)
        
        # Calculate yield with realistic factors
        yield_val = yield_base
        yield_val *= (1 - abs(temp - 25)/50)  # Temperature factor
        yield_val *= (1 - abs(rain - 1000)/2000)  # Rainfall factor
        yield_val *= soil_moisture/60  # Moisture factor
        yield_val *= (1 - abs(soil_ph - 6.5)/2)  # pH factor
        yield_val *= (N/100 + P/50 + K/80)/3  # Nutrient factor
        
        # Add noise
        yield_val += np.random.normal(0, yield_val * 0.15)
        yield_val = max(500, yield_val)
        
        data_yield.append([
            crop, round(temp,1), round(rain,1), round(soil_moisture,1),
            round(soil_ph,2), round(N,1), round(P,1), round(K,1),
            round(air_temp,1), round(air_moisture,1), round(soil_temp,1),
            round(sunlight,1), round(yield_val,2)
        ])

df_yield = pd.DataFrame(data_yield, columns=[
    'crop', 'temperature', 'rainfall', 'soil_moisture', 'soil_ph',
    'nitrogen', 'phosphorus', 'potassium', 'air_temperature',
    'air_moisture', 'soil_temperature', 'sunlight_hours', 'yield'
])

df_yield.to_csv('data/crop_yield_data.csv', index=False)
print(f" Created yield dataset: {len(df_yield)} samples, {len(crops)} crops")

# 2. Recommendation Dataset
print("\n📊 Generating recommendation dataset...")

crops_lower = [c.lower() for c in crops]
data_rec = []
for crop in crops_lower:
    for _ in range(1000):
        if crop == 'rice':
            N, P, K = 90, 42, 43
            temp, humid = 26.3, 82
            rain = 202.9
        elif crop == 'wheat':
            N, P, K = 80, 35, 40
            temp, humid = 20.5, 70
            rain = 80.5
        elif crop == 'apple':
            N, P, K = 100, 45, 60
            temp, humid = 18.5, 75
            rain = 150.5
        else:
            N, P, K = np.random.randint(70,110), np.random.randint(35,55), np.random.randint(45,75)
            temp = np.random.uniform(18,28)
            humid = np.random.uniform(65,85)
            rain = np.random.uniform(100,250)
        
        # Add variation
        N += np.random.randint(-10,10)
        P += np.random.randint(-5,5)
        K += np.random.randint(-5,5)
        temp += np.random.uniform(-2,2)
        humid += np.random.uniform(-5,5)
        rain += np.random.uniform(-20,20)
        ph = np.random.uniform(5.8,7.2)
        
        data_rec.append([
            max(0,N), max(0,P), max(0,K),
            round(temp,1), round(humid,1), round(ph,2),
            round(rain,1), crop
        ])

df_rec = pd.DataFrame(data_rec, columns=[
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'
])

df_rec.to_csv('data/crop_recommendation.csv', index=False)
print(f" Created recommendation dataset: {len(df_rec)} samples")

# 3. Soil Dataset
print("\n📊 Generating soil dataset...")

locations = ['Punjab', 'Maharashtra', 'Karnataka', 'Uttar Pradesh', 
             'West Bengal', 'Tamil Nadu', 'Rajasthan', 'Gujarat']
soil_types = ['Clay', 'Loam', 'Sandy', 'Silt', 'Peaty', 'Chalky']

data_soil = []
for location in locations:
    for _ in range(500):
        data_soil.append([
            location,
            np.random.uniform(20, 35),
            np.random.uniform(6.0, 8.0),
            np.random.uniform(30, 70),
            np.random.uniform(5.5, 7.5),
            np.random.randint(50, 120),
            np.random.randint(30, 70),
            np.random.randint(40, 90),
            np.random.choice(soil_types)
        ])

df_soil = pd.DataFrame(data_soil, columns=[
    'location', 'soil_temp', 'air_temp', 'air_moisture', 
    'soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'soil_type'
])

df_soil.to_csv('data/soil_data.csv', index=False)
print(f" Created soil dataset: {len(df_soil)} samples")

# 4. Train Yield Model
print("\n🤖 Training yield prediction model...")

X_yield = df_yield.drop('yield', axis=1)
y_yield = df_yield['yield']

le_yield = LabelEncoder()
X_yield_encoded = X_yield.copy()
X_yield_encoded['crop'] = le_yield.fit_transform(X_yield['crop'])

scaler_yield = StandardScaler()
X_yield_scaled = scaler_yield.fit_transform(X_yield_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_yield_scaled, y_yield, test_size=0.2, random_state=42
)

rf_yield = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_yield.fit(X_train, y_train)
y_pred = rf_yield.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f" Yield model trained. R² Score: {r2:.4f}")

# 5. Train Recommendation Model
print("\n🤖 Training crop recommendation model...")

X_rec = df_rec.drop('label', axis=1)
y_rec = df_rec['label']

le_rec = LabelEncoder()
y_rec_encoded = le_rec.fit_transform(y_rec)

scaler_rec = StandardScaler()
X_rec_scaled = scaler_rec.fit_transform(X_rec)

X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(
    X_rec_scaled, y_rec_encoded, test_size=0.2, random_state=42
)

rf_rec = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_rec.fit(X_train_rec, y_train_rec)
y_pred_rec = rf_rec.predict(X_test_rec)
acc = accuracy_score(y_test_rec, y_pred_rec)

print(f" Recommendation model trained. Accuracy: {acc:.4f}")

# 6. Save all models
print("\n💾 Saving models...")

joblib.dump(rf_yield, 'models/crop_yield_model.pkl')
joblib.dump(rf_rec, 'models/crop_recommendation_model.pkl')

label_encoders = {
    'crop_yield': le_yield,
    'crop_recommendation': le_rec
}
joblib.dump(label_encoders, 'models/label_encoders.pkl')

scalers = {
    'yield_scaler': scaler_yield,
    'recommendation_scaler': scaler_rec
}
joblib.dump(scalers, 'models/scaler.pkl')

print("\n" + "="*50)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*50)
print("\n📁 Files created:")
print(" data/crop_yield_data.csv")
print(" data/crop_recommendation.csv")
print(" data/soil_data.csv")
print(" models/crop_yield_model.pkl")
print(" models/crop_recommendation_model.pkl")
print(" models/label_encoders.pkl")
print(" models/scaler.pkl")
print(f"\n📊 Model Performance:")
print(f"   Yield Prediction R²: {r2:.4f}")
print(f"   Crop Recommendation Accuracy: {acc:.4f}")
print("\n🚀 Run the dashboard: streamlit run app.py")