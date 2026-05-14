import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Handles loading, processing, and managing agricultural data.
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize DataLoader with data and models directories.
        
        Args:
            data_dir: Directory containing data files
            models_dir: Directory containing model files
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self._create_directories()
        
        # Data containers
        self.yield_data = None
        self.recommendation_data = None
        self.soil_data = None
        self.geo_data = None
        
        # Model containers
        self.yield_model = None
        self.recommendation_model = None
        self.label_encoders = None
        self.scaler = None
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all agricultural datasets.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        datasets = {}
        
        try:
            # Load crop yield data
            self.yield_data = self.load_yield_data()
            datasets['yield_data'] = self.yield_data
            
            # Load crop recommendation data
            self.recommendation_data = self.load_recommendation_data()
            datasets['recommendation_data'] = self.recommendation_data
            
            # Load soil data
            self.soil_data = self.load_soil_data()
            datasets['soil_data'] = self.soil_data
            
            # Load geographic data (if exists)
            self.geo_data = self.load_geographic_data()
            if self.geo_data is not None:
                datasets['geo_data'] = self.geo_data
            
            print("✅ All data loaded successfully")
            return datasets
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def load_yield_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load crop yield data.
        
        Args:
            file_path: Path to yield data CSV file
            
        Returns:
            DataFrame containing yield data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "crop_yield_data.csv")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Yield data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Basic data validation
            required_columns = ['Crop', 'State', 'District', 'Season', 'Area', 'Yield']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing columns in yield data: {missing_cols}")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Yield data file not found: {file_path}")
            # Create sample data if file doesn't exist
            return self._create_sample_yield_data()
        except Exception as e:
            print(f"❌ Error loading yield data: {e}")
            raise
    
    def load_recommendation_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load crop recommendation data.
        
        Args:
            file_path: Path to recommendation data CSV file
            
        Returns:
            DataFrame containing recommendation data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "crop_recommendation.csv")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Recommendation data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Basic data validation
            required_columns = ['State', 'District', 'Season', 'N', 'P', 'K', 'ph', 'rainfall', 'Crop']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing columns in recommendation data: {missing_cols}")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Recommendation data file not found: {file_path}")
            # Create sample data if file doesn't exist
            return self._create_sample_recommendation_data()
        except Exception as e:
            print(f"❌ Error loading recommendation data: {e}")
            raise
    
    def load_soil_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load soil data.
        
        Args:
            file_path: Path to soil data CSV file
            
        Returns:
            DataFrame containing soil data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "soil_data.csv")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Soil data loaded: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except FileNotFoundError:
            print(f"❌ Soil data file not found: {file_path}")
            # Create sample data if file doesn't exist
            return self._create_sample_soil_data()
        except Exception as e:
            print(f"❌ Error loading soil data: {e}")
            raise
    
    def load_geographic_data(self, file_path: str = None) -> Optional[pd.DataFrame]:
        """
        Load geographic data with coordinates.
        
        Args:
            file_path: Path to geographic data CSV file
            
        Returns:
            DataFrame containing geographic data or None if not found
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "geo_data.csv")
        
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"✅ Geographic data loaded: {len(df)} rows")
                return df
            else:
                print("ℹ️ No geographic data file found")
                return None
                
        except Exception as e:
            print(f"❌ Error loading geographic data: {e}")
            return None
    
    def load_models(self) -> Dict[str, Any]:
        """
        Load trained ML models and preprocessing objects.
        
        Returns:
            Dictionary containing all loaded models
        """
        models = {}
        
        try:
            # Load yield prediction model
            yield_model_path = os.path.join(self.models_dir, "crop_yield_model.pkl")
            if os.path.exists(yield_model_path):
                with open(yield_model_path, 'rb') as f:
                    self.yield_model = pickle.load(f)
                models['yield_model'] = self.yield_model
                print("✅ Yield prediction model loaded")
            else:
                print("⚠️ Yield prediction model not found")
            
            # Load recommendation model
            rec_model_path = os.path.join(self.models_dir, "crop_recommendation_model.pkl")
            if os.path.exists(rec_model_path):
                with open(rec_model_path, 'rb') as f:
                    self.recommendation_model = pickle.load(f)
                models['recommendation_model'] = self.recommendation_model
                print("✅ Crop recommendation model loaded")
            else:
                print("⚠️ Crop recommendation model not found")
            
            # Load label encoders
            encoders_path = os.path.join(self.models_dir, "label_encoders.pkl")
            if os.path.exists(encoders_path):
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                models['label_encoders'] = self.label_encoders
                print("✅ Label encoders loaded")
            else:
                print("⚠️ Label encoders not found")
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                models['scaler'] = self.scaler
                print("✅ Scaler loaded")
            else:
                print("⚠️ Scaler not found")
            
            return models
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def get_crop_list(self) -> List[str]:
        """
        Get list of all available crops.
        
        Returns:
            List of crop names
        """
        if self.yield_data is not None and 'Crop' in self.yield_data.columns:
            return sorted(self.yield_data['Crop'].unique().tolist())
        elif self.recommendation_data is not None and 'Crop' in self.recommendation_data.columns:
            return sorted(self.recommendation_data['Crop'].unique().tolist())
        else:
            return []
    
    def get_state_list(self) -> List[str]:
        """
        Get list of all available states.
        
        Returns:
            List of state names
        """
        if self.yield_data is not None and 'State' in self.yield_data.columns:
            return sorted(self.yield_data['State'].unique().tolist())
        elif self.recommendation_data is not None and 'State' in self.recommendation_data.columns:
            return sorted(self.recommendation_data['State'].unique().tolist())
        else:
            return []
    
    def get_district_list(self, state: str = None) -> List[str]:
        """
        Get list of districts, optionally filtered by state.
        
        Args:
            state: State to filter districts by
            
        Returns:
            List of district names
        """
        if self.yield_data is not None and 'District' in self.yield_data.columns:
            if state:
                districts = self.yield_data[self.yield_data['State'] == state]['District'].unique()
            else:
                districts = self.yield_data['District'].unique()
            return sorted(districts.tolist())
        else:
            return []
    
    def get_crop_statistics(self, crop_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific crop.
        
        Args:
            crop_name: Name of the crop
            
        Returns:
            Dictionary containing crop statistics
        """
        if self.yield_data is None:
            return {}
        
        crop_data = self.yield_data[self.yield_data['Crop'] == crop_name]
        
        if crop_data.empty:
            return {"error": f"Crop '{crop_name}' not found"}
        
        stats = {
            "crop_name": crop_name,
            "total_samples": len(crop_data),
            "avg_yield": crop_data['Yield'].mean(),
            "max_yield": crop_data['Yield'].max(),
            "min_yield": crop_data['Yield'].min(),
            "std_yield": crop_data['Yield'].std(),
            "avg_area": crop_data['Area'].mean(),
            "avg_rainfall": crop_data['Rainfall'].mean() if 'Rainfall' in crop_data.columns else None,
            "avg_temperature": crop_data['Temperature'].mean() if 'Temperature' in crop_data.columns else None,
            "top_states": crop_data['State'].value_counts().head(3).to_dict(),
            "seasonal_distribution": crop_data['Season'].value_counts().to_dict()
        }
        
        return stats
    
    def get_soil_analysis(self, crop_name: str = None) -> Dict[str, Any]:
        """
        Get soil analysis data, optionally filtered by crop.
        
        Args:
            crop_name: Crop to filter by (optional)
            
        Returns:
            Dictionary containing soil analysis
        """
        if self.soil_data is None:
            return {}
        
        if crop_name and self.recommendation_data is not None:
            # Filter soil conditions for specific crop
            crop_recs = self.recommendation_data[self.recommendation_data['Crop'] == crop_name]
            if not crop_recs.empty:
                avg_soil = {
                    'N': crop_recs['N'].mean(),
                    'P': crop_recs['P'].mean(),
                    'K': crop_recs['K'].mean(),
                    'ph': crop_recs['ph'].mean(),
                    'rainfall': crop_recs['rainfall'].mean()
                }
            else:
                avg_soil = {}
        else:
            # Overall soil statistics
            avg_soil = {
                'pH': self.soil_data['pH'].mean(),
                'Organic_Carbon': self.soil_data['Organic_Carbon'].mean(),
                'Nitrogen': self.soil_data['Nitrogen'].mean(),
                'Phosphorus': self.soil_data['Phosphorus'].mean(),
                'Potassium': self.soil_data['Potassium'].mean(),
                'Moisture': self.soil_data['Moisture'].mean() if 'Moisture' in self.soil_data.columns else None
            }
        
        return avg_soil
    
    def _create_sample_yield_data(self) -> pd.DataFrame:
        """
        Create sample yield data for testing.
        
        Returns:
            DataFrame with sample yield data
        """
        print("📊 Creating sample yield data...")
        
        crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton']
        states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh']
        districts = ['Ludhiana', 'Hisar', 'Meerut', 'Indore']
        seasons = ['Kharif', 'Rabi']
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Crop': np.random.choice(crops, n_samples),
            'State': np.random.choice(states, n_samples),
            'District': np.random.choice(districts, n_samples),
            'Season': np.random.choice(seasons, n_samples),
            'Area': np.random.uniform(0.5, 50, n_samples),
            'Rainfall': np.random.uniform(500, 2000, n_samples),
            'Temperature': np.random.uniform(15, 35, n_samples),
            'Fertilizers': np.random.uniform(0, 200, n_samples),
            'Yield': np.random.uniform(1000, 5000, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save sample data
        output_path = os.path.join(self.data_dir, "crop_yield_data.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✅ Sample yield data created and saved to {output_path}")
        return df
    
    def _create_sample_recommendation_data(self) -> pd.DataFrame:
        """
        Create sample recommendation data for testing.
        
        Returns:
            DataFrame with sample recommendation data
        """
        print("📊 Creating sample recommendation data...")
        
        crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton']
        states = ['Punjab', 'Haryana', 'Uttar Pradesh']
        districts = ['Ludhiana', 'Hisar', 'Meerut']
        seasons = ['Kharif', 'Rabi']
        
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'State': np.random.choice(states, n_samples),
            'District': np.random.choice(districts, n_samples),
            'Season': np.random.choice(seasons, n_samples),
            'N': np.random.randint(0, 140, n_samples),
            'P': np.random.randint(0, 100, n_samples),
            'K': np.random.randint(0, 200, n_samples),
            'ph': np.random.uniform(5.0, 8.5, n_samples),
            'rainfall': np.random.randint(500, 1500, n_samples),
            'Crop': np.random.choice(crops, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save sample data
        output_path = os.path.join(self.data_dir, "crop_recommendation.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✅ Sample recommendation data created and saved to {output_path}")
        return df
    
    def _create_sample_soil_data(self) -> pd.DataFrame:
        """
        Create sample soil data for testing.
        
        Returns:
            DataFrame with sample soil data
        """
        print("📊 Creating sample soil data...")
        
        soil_types = ['Clay', 'Sandy', 'Loamy', 'Silty']
        
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'Soil_Type': np.random.choice(soil_types, n_samples),
            'pH': np.random.uniform(4.5, 8.5, n_samples),
            'Organic_Carbon': np.random.uniform(0.5, 3.0, n_samples),
            'Nitrogen': np.random.uniform(100, 500, n_samples),
            'Phosphorus': np.random.uniform(10, 100, n_samples),
            'Potassium': np.random.uniform(100, 600, n_samples),
            'Moisture': np.random.uniform(10, 50, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save sample data
        output_path = os.path.join(self.data_dir, "soil_data.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✅ Sample soil data created and saved to {output_path}")
        return df
    
    def generate_sample_geo_data(self) -> pd.DataFrame:
        """
        Generate sample geographic data with coordinates.
        
        Returns:
            DataFrame with geographic data
        """
        print("🗺️ Creating sample geographic data...")
        
        # Indian states with approximate coordinates
        geo_data = {
            'State': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh', 
                     'Rajasthan', 'Maharashtra', 'Karnataka', 'Andhra Pradesh'],
            'Latitude': [31.1471, 29.0588, 26.8467, 22.9734, 
                        27.0238, 19.7515, 15.3173, 15.9129],
            'Longitude': [75.3412, 76.0856, 80.9462, 78.6569, 
                         74.2179, 75.7139, 75.7139, 79.7400],
            'Capital': ['Chandigarh', 'Chandigarh', 'Lucknow', 'Bhopal',
                       'Jaipur', 'Mumbai', 'Bengaluru', 'Amaravati']
        }
        
        df = pd.DataFrame(geo_data)
        
        # Save geographic data
        output_path = os.path.join(self.data_dir, "geo_data.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✅ Sample geographic data created and saved to {output_path}")
        return df

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Get available crops and states
    crops = loader.get_crop_list()
    states = loader.get_state_list()
    
    print(f"\n📋 Available crops: {len(crops)}")
    print(f"📋 Available states: {len(states)}")
    
    # Get statistics for a crop
    if crops:
        crop_stats = loader.get_crop_statistics(crops[0])
        print(f"\n📊 Statistics for {crops[0]}:")
        for key, value in crop_stats.items():
            print(f"  {key}: {value}")
    
    # Get soil analysis
    soil_analysis = loader.get_soil_analysis()
    print(f"\n🧪 Soil Analysis:")
    for key, value in soil_analysis.items():
        print(f"  {key}: {value}")