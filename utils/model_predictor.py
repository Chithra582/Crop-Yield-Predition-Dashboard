import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

class ModelPredictor:
    """
    Handles predictions using trained ML models for crop yield and recommendations.
    """
    
    def __init__(self, data_loader=None, models: Dict[str, Any] = None):
        """
        Initialize ModelPredictor with models.
        
        Args:
            data_loader: DataLoader instance (optional)
            models: Dictionary containing pre-loaded models (optional)
        """
        self.data_loader = data_loader
        self.models = models or {}
        
        # Model references
        self.yield_model = None
        self.recommendation_model = None
        self.label_encoders = None
        self.scaler = None
        
        # Load models if provided
        if models:
            self._load_models_from_dict(models)
    
    def _load_models_from_dict(self, models: Dict[str, Any]):
        """Load models from dictionary."""
        self.yield_model = models.get('yield_model')
        self.recommendation_model = models.get('recommendation_model')
        self.label_encoders = models.get('label_encoders')
        self.scaler = models.get('scaler')
    
    def predict_yield(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict crop yield based on input parameters.
        
        Args:
            input_data: Dictionary containing prediction parameters
            
        Returns:
            Dictionary containing prediction results and analysis
        """
        if self.yield_model is None:
            return {"error": "Yield prediction model not loaded"}
        
        try:
            # Validate required parameters
            required_params = ['Crop', 'State', 'District', 'Season', 'Area']
            missing_params = [p for p in required_params if p not in input_data]
            if missing_params:
                return {"error": f"Missing required parameters: {missing_params}"}
            
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input data
            processed_input = self._preprocess_yield_input(input_df)
            
            # Make prediction
            prediction = self.yield_model.predict(processed_input)[0]
            
            # Get confidence score (if available)
            confidence = self._get_prediction_confidence(processed_input, model_type='yield')
            
            # Get historical comparison
            comparison = self._get_historical_comparison(input_data, prediction)
            
            # Generate recommendations
            recommendations = self._generate_yield_recommendations(input_data, prediction)
            
            # Prepare result
            result = {
                "predicted_yield": float(prediction),
                "prediction_units": "kg/ha",
                "confidence_score": confidence,
                "input_parameters": input_data,
                "historical_comparison": comparison,
                "recommendations": recommendations,
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}", "success": False}
    
    def recommend_crop(self, input_data: Dict[str, Any], top_n: int = 3) -> Dict[str, Any]:
        """
        Recommend suitable crops based on input parameters.
        
        Args:
            input_data: Dictionary containing recommendation parameters
            top_n: Number of top recommendations to return
            
        Returns:
            Dictionary containing crop recommendations
        """
        if self.recommendation_model is None:
            return {"error": "Crop recommendation model not loaded"}
        
        try:
            # Validate required parameters
            required_params = ['State', 'District', 'Season', 'N', 'P', 'K', 'ph', 'rainfall']
            missing_params = [p for p in required_params if p not in input_data]
            if missing_params:
                return {"error": f"Missing required parameters: {missing_params}"}
            
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input data
            processed_input = self._preprocess_recommendation_input(input_df)
            
            # Get prediction probabilities for all crops
            if hasattr(self.recommendation_model, 'predict_proba'):
                probabilities = self.recommendation_model.predict_proba(processed_input)[0]
                classes = self.recommendation_model.classes_
                
                # Sort by probability
                sorted_indices = np.argsort(probabilities)[::-1]
                top_indices = sorted_indices[:top_n]
                
                recommendations = []
                for idx in top_indices:
                    crop_name = self.label_encoders['Crop'].inverse_transform([classes[idx]])[0] \
                               if 'Crop' in self.label_encoders else f"Crop_{classes[idx]}"
                    
                    recommendations.append({
                        "crop": crop_name,
                        "probability": float(probabilities[idx]),
                        "suitability_score": float(probabilities[idx] * 100)
                    })
                
                # Get top prediction
                top_crop_idx = sorted_indices[0]
                top_crop = self.label_encoders['Crop'].inverse_transform([classes[top_crop_idx]])[0] \
                          if 'Crop' in self.label_encoders else f"Crop_{classes[top_crop_idx]}"
                
            else:
                # Model doesn't support probabilities, just get top prediction
                prediction = self.recommendation_model.predict(processed_input)[0]
                top_crop = self.label_encoders['Crop'].inverse_transform([prediction])[0] \
                          if 'Crop' in self.label_encoders else f"Crop_{prediction}"
                
                recommendations = [{
                    "crop": top_crop,
                    "probability": 1.0,
                    "suitability_score": 100.0
                }]
            
            # Get crop details
            crop_details = self._get_crop_details(top_crop, input_data)
            
            # Generate planting advice
            planting_advice = self._generate_planting_advice(top_crop, input_data)
            
            # Prepare result
            result = {
                "top_recommendation": top_crop,
                "recommendations": recommendations,
                "crop_details": crop_details,
                "planting_advice": planting_advice,
                "input_parameters": input_data,
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Recommendation failed: {str(e)}", "success": False}
    
    def analyze_soil(self, soil_parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze soil quality and provide recommendations.
        
        Args:
            soil_parameters: Dictionary containing soil parameters
            
        Returns:
            Dictionary containing soil analysis results
        """
        try:
            # Extract parameters with defaults
            n = soil_parameters.get('N', 0)
            p = soil_parameters.get('P', 0)
            k = soil_parameters.get('K', 0)
            ph = soil_parameters.get('pH', 7.0)
            organic_carbon = soil_parameters.get('Organic_Carbon', 1.0)
            
            # Calculate soil health score
            health_score, health_category = self._calculate_soil_health_score(n, p, k, ph, organic_carbon)
            
            # Analyze individual parameters
            parameter_analysis = self._analyze_soil_parameters(n, p, k, ph, organic_carbon)
            
            # Generate recommendations
            recommendations = self._generate_soil_recommendations(parameter_analysis)
            
            # Prepare result
            result = {
                "soil_health_score": health_score,
                "soil_health_category": health_category,
                "parameter_analysis": parameter_analysis,
                "recommendations": recommendations,
                "input_parameters": soil_parameters,
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Soil analysis failed: {str(e)}", "success": False}
    
    def predict_multiple_yields(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict yields for multiple scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            
        Returns:
            Dictionary containing predictions for all scenarios
        """
        results = []
        
        for i, scenario in enumerate(scenarios):
            prediction = self.predict_yield(scenario)
            results.append({
                "scenario_id": i + 1,
                "scenario": scenario,
                "prediction": prediction
            })
        
        # Calculate statistics
        successful_predictions = [r for r in results if r["prediction"].get("success", False)]
        
        if successful_predictions:
            yields = [r["prediction"]["predicted_yield"] for r in successful_predictions]
            
            stats = {
                "average_yield": np.mean(yields),
                "max_yield": np.max(yields),
                "min_yield": np.min(yields),
                "std_yield": np.std(yields),
                "best_scenario": successful_predictions[np.argmax(yields)]["scenario_id"],
                "worst_scenario": successful_predictions[np.argmin(yields)]["scenario_id"]
            }
        else:
            stats = {}
        
        return {
            "scenario_results": results,
            "statistics": stats,
            "total_scenarios": len(scenarios),
            "successful_predictions": len(successful_predictions)
        }
    
    def optimize_parameters(self, base_scenario: Dict[str, Any],
                           parameter_ranges: Dict[str, Tuple[float, float]],
                           steps: int = 5) -> Dict[str, Any]:
        """
        Optimize input parameters to maximize yield.
        
        Args:
            base_scenario: Base scenario dictionary
            parameter_ranges: Dictionary of parameter ranges to optimize
            steps: Number of steps per parameter
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Generate parameter combinations
            parameter_combinations = self._generate_parameter_combinations(
                base_scenario, parameter_ranges, steps
            )
            
            # Predict yields for all combinations
            predictions = []
            for combo in parameter_combinations:
                prediction = self.predict_yield(combo)
                if prediction.get("success", False):
                    predictions.append({
                        "parameters": combo,
                        "yield": prediction["predicted_yield"]
                    })
            
            if not predictions:
                return {"error": "No successful predictions", "success": False}
            
            # Find optimal parameters
            best_prediction = max(predictions, key=lambda x: x["yield"])
            worst_prediction = min(predictions, key=lambda x: x["yield"])
            
            # Calculate improvements
            base_prediction = self.predict_yield(base_scenario)
            if base_prediction.get("success", False):
                base_yield = base_prediction["predicted_yield"]
                improvement = ((best_prediction["yield"] - base_yield) / base_yield) * 100
            else:
                base_yield = None
                improvement = None
            
            # Prepare optimization insights
            insights = self._generate_optimization_insights(predictions, parameter_ranges)
            
            return {
                "optimal_parameters": best_prediction["parameters"],
                "optimal_yield": best_prediction["yield"],
                "worst_parameters": worst_prediction["parameters"],
                "worst_yield": worst_prediction["yield"],
                "base_yield": base_yield,
                "improvement_percentage": improvement,
                "total_combinations_tested": len(predictions),
                "insights": insights,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}", "success": False}
    
    def _preprocess_yield_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for yield prediction."""
        processed_df = input_df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Crop', 'State', 'District', 'Season']
        for col in categorical_cols:
            if col in processed_df.columns and col in self.label_encoders:
                try:
                    processed_df[col] = self.label_encoders[col].transform(processed_df[col])
                except:
                    # If category not in encoder, use unknown encoding
                    processed_df[col] = 0
        
        # Scale numerical features
        numerical_cols = ['Area', 'Rainfall', 'Temperature', 'Fertilizers']
        numerical_cols = [col for col in numerical_cols if col in processed_df.columns]
        
        if numerical_cols and self.scaler:
            processed_df[numerical_cols] = self.scaler.transform(processed_df[numerical_cols])
        
        # Ensure all required columns are present
        expected_features = self.yield_model.feature_names_in_ if hasattr(self.yield_model, 'feature_names_in_') else None
        if expected_features is not None:
            for feature in expected_features:
                if feature not in processed_df.columns:
                    processed_df[feature] = 0
        
        return processed_df
    
    def _preprocess_recommendation_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for crop recommendation."""
        processed_df = input_df.copy()
        
        # Encode categorical variables
        categorical_cols = ['State', 'District', 'Season']
        for col in categorical_cols:
            if col in processed_df.columns and col in self.label_encoders:
                try:
                    processed_df[col] = self.label_encoders[col].transform(processed_df[col])
                except:
                    processed_df[col] = 0
        
        return processed_df
    
    def _get_prediction_confidence(self, processed_input: pd.DataFrame, 
                                  model_type: str = 'yield') -> float:
        """Calculate prediction confidence score."""
        # Simple confidence estimation
        # In a real application, this could use prediction probabilities or ensemble variance
        
        if model_type == 'yield' and hasattr(self.yield_model, 'predict_proba'):
            # For regression, we might use prediction intervals
            return 0.85  # Placeholder
        elif model_type == 'recommendation' and hasattr(self.recommendation_model, 'predict_proba'):
            # Use the highest probability as confidence
            probabilities = self.recommendation_model.predict_proba(processed_input)[0]
            return float(np.max(probabilities))
        else:
            return 0.80  # Default confidence
    
    def _get_historical_comparison(self, input_data: Dict[str, Any], 
                                  predicted_yield: float) -> Dict[str, Any]:
        """Compare prediction with historical data."""
        if self.data_loader is None or self.data_loader.yield_data is None:
            return {}
        
        try:
            crop = input_data.get('Crop')
            state = input_data.get('State')
            district = input_data.get('District')
            season = input_data.get('Season')
            
            # Filter historical data
            historical = self.data_loader.yield_data.copy()
            
            if crop:
                historical = historical[historical['Crop'] == crop]
            if state:
                historical = historical[historical['State'] == state]
            if district:
                historical = historical[historical['District'] == district]
            if season:
                historical = historical[historical['Season'] == season]
            
            if historical.empty:
                return {"message": "No historical data for comparison"}
            
            # Calculate statistics
            avg_yield = historical['Yield'].mean()
            max_yield = historical['Yield'].max()
            min_yield = historical['Yield'].min()
            
            # Calculate difference
            diff = predicted_yield - avg_yield
            diff_percent = (diff / avg_yield) * 100 if avg_yield > 0 else 0
            
            return {
                "historical_average": float(avg_yield),
                "historical_maximum": float(max_yield),
                "historical_minimum": float(min_yield),
                "difference_from_average": float(diff),
                "difference_percentage": float(diff_percent),
                "comparison": "above average" if diff > 0 else "below average" if diff < 0 else "equal to average"
            }
            
        except:
            return {}
    
    def _generate_yield_recommendations(self, input_data: Dict[str, Any], 
                                       predicted_yield: float) -> List[Dict[str, str]]:
        """Generate recommendations to improve yield."""
        recommendations = []
        
        # Area-based recommendations
        area = input_data.get('Area', 0)
        if area < 1:
            recommendations.append({
                "category": "Area",
                "recommendation": "Consider increasing cultivation area for better economies of scale",
                "priority": "medium"
            })
        
        # Fertilizer recommendations
        fertilizers = input_data.get('Fertilizers', 0)
        if fertilizers < 50:
            recommendations.append({
                "category": "Fertilizer",
                "recommendation": "Increase fertilizer application for better nutrient availability",
                "priority": "high"
            })
        elif fertilizers > 200:
            recommendations.append({
                "category": "Fertilizer",
                "recommendation": "Reduce fertilizer usage to prevent nutrient leaching and save costs",
                "priority": "medium"
            })
        
        # Rainfall recommendations
        rainfall = input_data.get('Rainfall', 0)
        crop = input_data.get('Crop', '')
        
        if crop.lower() == 'rice' and rainfall < 1000:
            recommendations.append({
                "category": "Irrigation",
                "recommendation": "Rice requires more water. Consider supplementary irrigation",
                "priority": "high"
            })
        
        # General best practices
        recommendations.append({
            "category": "Best Practices",
            "recommendation": "Implement crop rotation and soil conservation practices",
            "priority": "medium"
        })
        
        return recommendations
    
    def _get_crop_details(self, crop_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about a crop."""
        if self.data_loader is None:
            return {}
        
        try:
            # Get crop statistics
            stats = self.data_loader.get_crop_statistics(crop_name)
            
            # Get soil requirements
            soil_req = self.data_loader.get_soil_analysis(crop_name)
            
            # Calculate suitability based on input
            suitability = self._calculate_crop_suitability(crop_name, input_data)
            
            return {
                "crop_name": crop_name,
                "statistics": stats,
                "soil_requirements": soil_req,
                "suitability_score": suitability
            }
            
        except:
            return {"crop_name": crop_name, "details": "Information not available"}
    
    def _calculate_crop_suitability(self, crop_name: str, input_data: Dict[str, Any]) -> float:
        """Calculate suitability score for a crop based on input conditions."""
        # Simple suitability calculation
        # In real application, this would use more sophisticated logic
        
        score = 0.7  # Base score
        
        # Adjust based on rainfall
        rainfall = input_data.get('rainfall', 0)
        if 800 <= rainfall <= 1200:
            score += 0.1
        elif 500 <= rainfall <= 1500:
            score += 0.05
        
        # Adjust based on soil pH
        ph = input_data.get('ph', 7.0)
        if 6.0 <= ph <= 7.5:
            score += 0.1
        elif 5.5 <= ph <= 8.0:
            score += 0.05
        
        return min(score, 1.0)
    
    def _generate_planting_advice(self, crop_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate planting and cultivation advice."""
        # This is a simplified version - in real app, would use detailed crop knowledge base
        
        advice = {
            "planting_season": self._get_planting_season(crop_name, input_data.get('Season')),
            "spacing": self._get_plant_spacing(crop_name),
            "irrigation": self._get_irrigation_advice(crop_name, input_data.get('rainfall')),
            "fertilization": self._get_fertilization_advice(crop_name),
            "pest_management": "Monitor regularly for pests and diseases",
            "harvest_time": "120-150 days after planting for most crops"
        }
        
        return advice
    
    def _get_planting_season(self, crop_name: str, season: str = None) -> str:
        """Get planting season for a crop."""
        season_mapping = {
            'Rice': 'Kharif (June-July)',
            'Wheat': 'Rabi (October-November)',
            'Maize': 'Kharif (June-July) or Rabi (October-November)',
            'Sugarcane': 'Throughout the year',
            'Cotton': 'Kharif (May-June)'
        }
        
        if crop_name in season_mapping:
            return season_mapping[crop_name]
        elif season:
            return f"{season} season"
        else:
            return "Depends on local climate"
    
    def _get_plant_spacing(self, crop_name: str) -> str:
        """Get recommended plant spacing."""
        spacing = {
            'Rice': '20x15 cm',
            'Wheat': '22.5 cm row spacing',
            'Maize': '60x20 cm',
            'Sugarcane': '90-120 cm row spacing',
            'Cotton': '90x60 cm'
        }
        
        return spacing.get(crop_name, "Consult local agricultural extension")
    
    def _get_irrigation_advice(self, crop_name: str, rainfall: float = 0) -> str:
        """Get irrigation advice."""
        if rainfall > 1200:
            return "Rainfall is sufficient. Irrigation may not be needed."
        elif rainfall > 800:
            return "Moderate irrigation may be needed during dry spells."
        else:
            return "Regular irrigation required for optimal growth."
    
    def _get_fertilization_advice(self, crop_name: str) -> str:
        """Get fertilization advice."""
        advice = {
            'Rice': '120:60:40 kg/ha NPK',
            'Wheat': '100:50:40 kg/ha NPK',
            'Maize': '120:60:40 kg/ha NPK',
            'Sugarcane': '200:80:160 kg/ha NPK',
            'Cotton': '80:40:40 kg/ha NPK'
        }
        
        return advice.get(crop_name, "Balanced NPK fertilizer recommended")
    
    def _calculate_soil_health_score(self, n: float, p: float, k: float, 
                                    ph: float, organic_carbon: float) -> Tuple[float, str]:
        """Calculate soil health score."""
        # Normalize parameters
        n_score = min(n / 200, 1.0)
        p_score = min(p / 100, 1.0)
        k_score = min(k / 300, 1.0)
        ph_score = 1 - abs(ph - 6.5) / 3.5  # Ideal pH is 6.5
        oc_score = min(organic_carbon / 3.0, 1.0)
        
        # Weighted average
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        scores = [n_score, p_score, k_score, ph_score, oc_score]
        
        total_score = sum(w * s for w, s in zip(weights, scores))
        
        # Categorize
        if total_score >= 0.8:
            category = "Excellent"
        elif total_score >= 0.6:
            category = "Good"
        elif total_score >= 0.4:
            category = "Fair"
        else:
            category = "Poor"
        
        return total_score, category
    
    def _analyze_soil_parameters(self, n: float, p: float, k: float, 
                                ph: float, organic_carbon: float) -> Dict[str, Dict[str, Any]]:
        """Analyze individual soil parameters."""
        analysis = {}
        
        # Nitrogen analysis
        if n < 100:
            n_status = "Low"
            n_advice = "Add nitrogen-rich fertilizers"
        elif n < 200:
            n_status = "Adequate"
            n_advice = "Maintain current levels"
        else:
            n_status = "High"
            n_advice = "Reduce nitrogen application"
        
        analysis['Nitrogen'] = {
            "value": n,
            "status": n_status,
            "optimal_range": "100-200 kg/ha",
            "advice": n_advice
        }
        
        # Phosphorus analysis
        if p < 30:
            p_status = "Low"
            p_advice = "Add phosphorus fertilizers"
        elif p < 60:
            p_status = "Adequate"
            p_advice = "Maintain current levels"
        else:
            p_status = "High"
            p_advice = "Reduce phosphorus application"
        
        analysis['Phosphorus'] = {
            "value": p,
            "status": p_status,
            "optimal_range": "30-60 kg/ha",
            "advice": p_advice
        }
        
        # Potassium analysis
        if k < 150:
            k_status = "Low"
            k_advice = "Add potassium fertilizers"
        elif k < 300:
            k_status = "Adequate"
            k_advice = "Maintain current levels"
        else:
            k_status = "High"
            k_advice = "Reduce potassium application"
        
        analysis['Potassium'] = {
            "value": k,
            "status": k_status,
            "optimal_range": "150-300 kg/ha",
            "advice": k_advice
        }
        
        # pH analysis
        if ph < 5.5:
            ph_status = "Acidic"
            ph_advice = "Add lime to raise pH"
        elif ph < 6.5:
            ph_status = "Slightly Acidic"
            ph_advice = "Good for most crops"
        elif ph < 7.5:
            ph_status = "Neutral"
            ph_advice = "Ideal for most crops"
        else:
            ph_status = "Alkaline"
            ph_advice = "Add sulfur to lower pH"
        
        analysis['pH'] = {
            "value": ph,
            "status": ph_status,
            "optimal_range": "6.0-7.5",
            "advice": ph_advice
        }
        
        # Organic Carbon analysis
        if organic_carbon < 0.8:
            oc_status = "Low"
            oc_advice = "Add organic matter/compost"
        elif organic_carbon < 1.5:
            oc_status = "Medium"
            oc_advice = "Add some organic matter"
        else:
            oc_status = "Good"
            oc_advice = "Maintain organic matter levels"
        
        analysis['Organic_Carbon'] = {
            "value": organic_carbon,
            "status": oc_status,
            "optimal_range": ">0.8%",
            "advice": oc_advice
        }
        
        return analysis
    
    def _generate_soil_recommendations(self, parameter_analysis: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate soil improvement recommendations."""
        recommendations = []
        
        for param, analysis in parameter_analysis.items():
            if analysis['status'] in ['Low', 'High']:
                recommendations.append({
                    "parameter": param,
                    "issue": f"{param} level is {analysis['status'].lower()}",
                    "recommendation": analysis['advice'],
                    "priority": "high" if analysis['status'] == 'Low' else "medium"
                })
        
        # General recommendations
        recommendations.append({
            "parameter": "General",
            "issue": "Soil health maintenance",
            "recommendation": "Add compost or farmyard manure annually",
            "priority": "medium"
        })
        
        return recommendations
    
    def _generate_parameter_combinations(self, base_scenario: Dict[str, Any],
                                        parameter_ranges: Dict[str, Tuple[float, float]],
                                        steps: int) -> List[Dict[str, Any]]:
        """Generate combinations of parameters for optimization."""
        combinations = []
        
        # Create parameter grids
        param_grids = {}
        for param, (min_val, max_val) in parameter_ranges.items():
            if steps > 1:
                param_grids[param] = np.linspace(min_val, max_val, steps)
            else:
                param_grids[param] = [min_val]
        
        # Generate combinations using recursion
        def generate_recursive(current_params, remaining_params):
            if not remaining_params:
                # Combine with base scenario
                full_params = base_scenario.copy()
                full_params.update(current_params)
                combinations.append(full_params)
                return
            
            param, values = remaining_params[0]
            for value in values:
                current_params[param] = value
                generate_recursive(current_params.copy(), remaining_params[1:])
        
        # Convert to list for easier processing
        param_list = list(param_grids.items())
        generate_recursive({}, param_list)
        
        return combinations
    
    def _generate_optimization_insights(self, predictions: List[Dict[str, Any]],
                                       parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Generate insights from optimization results."""
        if not predictions:
            return {}
        
        # Calculate parameter sensitivity
        sensitivities = {}
        for param in parameter_ranges.keys():
            param_values = [p["parameters"][param] for p in predictions]
            yields = [p["yield"] for p in predictions]
            
            # Simple correlation (could be improved)
            if len(set(param_values)) > 1:
                correlation = np.corrcoef(param_values, yields)[0, 1]
                sensitivities[param] = {
                    "correlation": float(correlation),
                    "effect": "positive" if correlation > 0 else "negative" if correlation < 0 else "neutral"
                }
        
        # Find patterns
        top_10 = sorted(predictions, key=lambda x: x["yield"], reverse=True)[:10]
        bottom_10 = sorted(predictions, key=lambda x: x["yield"])[:10]
        
        # Compare parameter distributions
        insights = {
            "parameter_sensitivities": sensitivities,
            "optimal_range_insights": self._find_optimal_ranges(top_10, parameter_ranges),
            "avoid_range_insights": self._find_avoid_ranges(bottom_10, parameter_ranges),
            "total_predictions": len(predictions),
            "yield_range": {
                "min": min(p["yield"] for p in predictions),
                "max": max(p["yield"] for p in predictions),
                "average": np.mean([p["yield"] for p in predictions])
            }
        }
        
        return insights
    
    def _find_optimal_ranges(self, top_predictions: List[Dict[str, Any]],
                            parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Find optimal parameter ranges from top predictions."""
        optimal_ranges = {}
        
        for param in parameter_ranges.keys():
            values = [p["parameters"][param] for p in top_predictions]
            if values:
                optimal_ranges[param] = {
                    "min": min(values),
                    "max": max(values),
                    "average": np.mean(values),
                    "recommendation": f"Aim for {min(values):.1f}-{max(values):.1f}"
                }
        
        return optimal_ranges
    
    def _find_avoid_ranges(self, bottom_predictions: List[Dict[str, Any]],
                          parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Find parameter ranges to avoid from bottom predictions."""
        avoid_ranges = {}
        
        for param in parameter_ranges.keys():
            values = [p["parameters"][param] for p in bottom_predictions]
            if values:
                avoid_ranges[param] = {
                    "min": min(values),
                    "max": max(values),
                    "average": np.mean(values),
                    "warning": f"Avoid {min(values):.1f}-{max(values):.1f} for low yields"
                }
        
        return avoid_ranges

# Example usage
if __name__ == "__main__":
    # Create a mock model predictor for testing
    predictor = ModelPredictor()
    
    # Example yield prediction
    yield_input = {
        'Crop': 'Rice',
        'State': 'Punjab',
        'District': 'Ludhiana',
        'Season': 'Kharif',
        'Area': 5.0,
        'Rainfall': 1200,
        'Temperature': 28,
        'Fertilizers': 100
    }
    
    print("🌾 Example Yield Prediction:")
    yield_result = predictor.predict_yield(yield_input)
    print(f"Result: {yield_result}")
    
    # Example crop recommendation
    rec_input = {
        'State': 'Punjab',
        'District': 'Ludhiana',
        'Season': 'Kharif',
        'N': 80,
        'P': 40,
        'K': 60,
        'ph': 6.8,
        'rainfall': 1200
    }
    
    print("\n🌱 Example Crop Recommendation:")
    rec_result = predictor.recommend_crop(rec_input, top_n=3)
    print(f"Result: {rec_result}")
    
    # Example soil analysis
    soil_input = {
        'N': 120,
        'P': 35,
        'K': 180,
        'pH': 6.5,
        'Organic_Carbon': 1.2
    }
    
    print("\n🧪 Example Soil Analysis:")
    soil_result = predictor.analyze_soil(soil_input)
    print(f"Result: {soil_result}")
    
    print("\n✅ Model predictor ready for use!")