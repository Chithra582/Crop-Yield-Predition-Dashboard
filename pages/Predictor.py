# pages/2_📈_Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Yield Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-header {
        background: linear-gradient(90deg, #FF9800, #FF5722);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .input-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .result-card {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .feature-card {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    
    .stSlider > div > div > div {
        background: #4CAF50 !important;
    }
    
    .prediction-tabs {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load sample model (simulated)
@st.cache_resource
def load_model():
    """Load or simulate ML model"""
    try:
        # In production, load actual model
        # model = joblib.load('models/crop_yield_model.pkl')
        return None
    except:
        return None

# Load crop data
@st.cache_data
def load_crop_data():
    """Load crop database"""
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 
             'Potato', 'Tomato', 'Apple', 'Banana', 'Grapes']
    
    crop_info = {
        'Rice': {'temp_range': (22, 32), 'rain_range': (1000, 2500), 'ph_range': (5.5, 6.5)},
        'Wheat': {'temp_range': (15, 25), 'rain_range': (300, 600), 'ph_range': (6.0, 7.5)},
        'Maize': {'temp_range': (18, 27), 'rain_range': (500, 800), 'ph_range': (5.8, 7.0)},
        'Cotton': {'temp_range': (25, 35), 'rain_range': (500, 800), 'ph_range': (6.0, 8.0)},
        'Sugarcane': {'temp_range': (20, 30), 'rain_range': (1500, 2500), 'ph_range': (6.5, 7.5)}
    }
    
    return crops, crop_info

# Load data
model = load_model()
crops, crop_info = load_crop_data()

# Initialize session state for predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main Predictor Content
st.markdown('<h1 class="prediction-header">📈 CROP YIELD PREDICTOR</h1>', unsafe_allow_html=True)
st.markdown("### AI-powered Yield Prediction with Machine Learning")

# Create tabs for different prediction types
pred_tabs = st.tabs(["🌾 Yield Prediction", "🤖 Crop Recommendation", "📊 Compare Crops"])

with pred_tabs[0]:  # Yield Prediction
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Input Parameters")
        
        with st.form("yield_prediction_form"):
            # Location selection
            location = st.selectbox(
                "📍 Location",
                ['Punjab', 'Maharashtra', 'Uttar Pradesh', 'Karnataka', 
                 'West Bengal', 'Tamil Nadu', 'Rajasthan', 'Gujarat']
            )
            
            # Crop selection with info
            selected_crop = st.selectbox(
                "🌾 Select Crop",
                crops,
                help="Choose the crop for yield prediction"
            )
            
            # Show crop info
            if selected_crop in crop_info:
                info = crop_info[selected_crop]
                st.caption(f"**Optimal Range:** Temp: {info['temp_range'][0]}-{info['temp_range'][1]}°C, "
                          f"Rain: {info['rain_range'][0]}-{info['rain_range'][1]} mm, "
                          f"pH: {info['ph_range'][0]}-{info['ph_range'][1]}")
            
            st.markdown("---")
            
            # Climate parameters
            st.markdown("#### 🌦️ Climate Parameters")
            
            temp_col, rain_col = st.columns(2)
            with temp_col:
                temperature = st.slider(
                    "Temperature (°C)",
                    10.0, 40.0, 25.0, 0.1,
                    help="Average daily temperature"
                )
            
            with rain_col:
                rainfall = st.slider(
                    "Rainfall (mm)",
                    200.0, 2500.0, 800.0, 10.0,
                    help="Annual rainfall in millimeters"
                )
            
            # Soil parameters
            st.markdown("#### 🌱 Soil Parameters")
            
            soil_col1, soil_col2 = st.columns(2)
            with soil_col1:
                soil_moisture = st.slider(
                    "Soil Moisture (%)",
                    20.0, 80.0, 50.0, 1.0,
                    help="Percentage of water in soil"
                )
            
            with soil_col2:
                soil_ph = st.slider(
                    "Soil pH",
                    5.0, 8.5, 6.5, 0.1,
                    help="Acidity/Alkalinity level (7 is neutral)"
                )
            
            soil_temp = st.slider(
                "Soil Temperature (°C)",
                10.0, 40.0, 22.0, 0.1,
                help="Temperature at root zone"
            )
            
            # Nutrient parameters
            st.markdown("#### 🧪 Nutrient Levels (kg/ha)")
            
            n_col, p_col, k_col = st.columns(3)
            with n_col:
                nitrogen = st.number_input(
                    "Nitrogen (N)",
                    0, 200, 100,
                    help="Essential for leaf growth"
                )
            
            with p_col:
                phosphorus = st.number_input(
                    "Phosphorus (P)",
                    0, 150, 50,
                    help="Important for root development"
                )
            
            with k_col:
                potassium = st.number_input(
                    "Potassium (K)",
                    0, 200, 60,
                    help="Helps with disease resistance"
                )
            
            # Additional parameters
            st.markdown("#### ☀️ Additional Parameters")
            
            sunlight = st.slider(
                "Sunlight Hours",
                4.0, 12.0, 8.0, 0.5,
                help="Average daily sunlight exposure"
            )
            
            elevation = st.slider(
                "Elevation (meters)",
                0, 2000, 200, 10,
                help="Altitude above sea level"
            )
            
            soil_type = st.selectbox(
                "🏜️ Soil Type",
                ['Loamy', 'Clayey', 'Sandy', 'Silty', 'Peaty', 'Chalky']
            )
            
            # Submit button
            submit = st.form_submit_button(
                "🚀 PREDICT YIELD",
                type="primary",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submit:
            # Simulate ML prediction (replace with actual model prediction)
            if selected_crop == 'Rice':
                base_yield = 4500
                temp_factor = 1 - abs(temperature - 28) / 20
                rain_factor = 1 - abs(rainfall - 1500) / 1000
            elif selected_crop == 'Wheat':
                base_yield = 3500
                temp_factor = 1 - abs(temperature - 20) / 15
                rain_factor = 1 - abs(rainfall - 500) / 500
            elif selected_crop == 'Apple':
                base_yield = 25000
                temp_factor = 1 - abs(temperature - 18) / 10
                rain_factor = 1 - abs(rainfall - 800) / 400
            else:
                base_yield = 4000
                temp_factor = 1 - abs(temperature - 25) / 15
                rain_factor = 1 - abs(rainfall - 1000) / 800
            
            # Calculate yield with factors
            prediction = base_yield * temp_factor * rain_factor
            prediction *= (soil_moisture / 60)  # Moisture factor
            prediction *= (1 - abs(soil_ph - 6.5) / 2)  # pH factor
            prediction *= (nitrogen/100 + phosphorus/50 + potassium/80) / 3  # Nutrient factor
            prediction *= (sunlight / 8)  # Sunlight factor
            
            # Add some variation
            prediction += np.random.normal(0, prediction * 0.1)
            prediction = max(1000, round(prediction))
            
            # Calculate confidence
            confidence = min(98, max(75, 
                90 - abs(temperature - 25) - abs(rainfall - 1000)/100 + 
                (soil_moisture - 40) + (nitrogen - 80)/2
            ))
            
            # Store prediction
            prediction_record = {
                'timestamp': datetime.now(),
                'crop': selected_crop,
                'location': location,
                'yield': prediction,
                'confidence': confidence,
                'parameters': {
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'soil_moisture': soil_moisture,
                    'soil_ph': soil_ph,
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium
                }
            }
            
            st.session_state.prediction_history.append(prediction_record)
            
            # Display results
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"## 🌟 {selected_crop.upper()}")
            st.markdown(f"# {prediction:,.0f} kg/ha")
            st.markdown(f"**📍 Location:** {location}")
            st.markdown(f"**🎯 Confidence:** {confidence:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Yield gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Yield"},
                gauge={
                    'axis': {'range': [None, 10000]},
                    'bar': {'color': "#4CAF50"},
                    'steps': [
                        {'range': [0, 3000], 'color': "#ff5252"},
                        {'range': [3000, 6000], 'color': "#ffd740"},
                        {'range': [6000, 10000], 'color': "#69f0ae"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 7000
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Parameter analysis
            st.markdown("#### 🔍 Parameter Analysis")
            
            parameters = {
                'Temperature': temperature,
                'Rainfall': rainfall,
                'Soil Moisture': soil_moisture,
                'Soil pH': soil_ph,
                'Nitrogen': nitrogen,
                'Phosphorus': phosphorus,
                'Potassium': potassium
            }
            
            optimal_ranges = {
                'Temperature': (20, 30),
                'Rainfall': (500, 1500),
                'Soil Moisture': (40, 60),
                'Soil pH': (6.0, 7.0),
                'Nitrogen': (80, 120),
                'Phosphorus': (40, 80),
                'Potassium': (50, 100)
            }
            
            analysis_data = []
            for param, value in parameters.items():
                opt_min, opt_max = optimal_ranges.get(param, (0, 100))
                status = "✅ Optimal" if opt_min <= value <= opt_max else "⚠️ Adjust"
                analysis_data.append({
                    'Parameter': param,
                    'Value': value,
                    'Optimal Range': f"{opt_min}-{opt_max}",
                    'Status': status
                })
            
            st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            
            recommendations = []
            if soil_moisture < 40:
                recommendations.append("💧 **Increase irrigation**: Soil moisture below optimal (40-60% recommended)")
            elif soil_moisture > 70:
                recommendations.append("☀️ **Improve drainage**: High moisture may cause root diseases")
            
            if nitrogen < 80:
                recommendations.append("🌱 **Apply nitrogen fertilizer**: Low nitrogen levels detected")
            elif nitrogen > 150:
                recommendations.append("⚠️ **Reduce nitrogen**: Excess can harm crop and environment")
            
            if soil_ph < 6.0:
                recommendations.append("⚗️ **Add lime**: Soil is too acidic for optimal growth")
            elif soil_ph > 7.5:
                recommendations.append("🧪 **Add sulfur**: Soil is too alkaline")
            
            if not recommendations:
                recommendations.append("✅ **All parameters in optimal range!** Maintain current practices.")
            
            for rec in recommendations:
                st.info(rec)
            
            # Export option
            st.markdown("#### 📥 Export Prediction")
            export_data = pd.DataFrame([{
                'Crop': selected_crop,
                'Location': location,
                'Predicted_Yield': prediction,
                'Confidence': confidence,
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M")
            }])
            
            st.download_button(
                label="📄 Download as CSV",
                data=export_data.to_csv(index=False),
                file_name=f"yield_prediction_{selected_crop}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:
            st.info("👈 Enter parameters on the left and click **PREDICT YIELD**")
            
            # Show sample predictions if available
            if st.session_state.prediction_history:
                st.markdown("#### 📋 Previous Predictions")
                recent_preds = st.session_state.prediction_history[-3:]
                for pred in reversed(recent_preds):
                    with st.expander(f"{pred['crop']} at {pred['location']} ({pred['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                        st.write(f"**Yield:** {pred['yield']:,} kg/ha")
                        st.write(f"**Confidence:** {pred['confidence']:.1f}%")

with pred_tabs[1]:  # Crop Recommendation
    st.markdown("## 🤖 Crop Recommendation System")
    st.markdown("### AI-based crop suggestion based on your conditions")
    
    rec_col1, rec_col2 = st.columns([1, 1])
    
    with rec_col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Input Soil & Climate Data")
        
        with st.form("recommendation_form"):
            # Soil nutrients
            st.markdown("#### 🧪 Soil Nutrients (kg/ha)")
            
            n_col, p_col, k_col = st.columns(3)
            with n_col:
                rec_nitrogen = st.slider("N", 0, 200, 90, key="rec_n")
            with p_col:
                rec_phosphorus = st.slider("P", 0, 150, 42, key="rec_p")
            with k_col:
                rec_potassium = st.slider("K", 0, 200, 43, key="rec_k")
            
            st.markdown("---")
            
            # Climate parameters
            st.markdown("#### 🌦️ Climate Parameters")
            
            rec_temp_col, rec_humid_col = st.columns(2)
            with rec_temp_col:
                rec_temperature = st.slider("Temperature (°C)", 10.0, 40.0, 26.3, 0.1, key="rec_temp")
            with rec_humid_col:
                rec_humidity = st.slider("Humidity (%)", 30.0, 100.0, 82.0, 0.1, key="rec_humid")
            
            rec_rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 202.9, 0.1, key="rec_rain")
            
            rec_ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1, key="rec_ph")
            
            rec_location = st.selectbox(
                "📍 Region",
                ['North India', 'South India', 'East India', 'West India', 'Central India']
            )
            
            rec_submit = st.form_submit_button(
                "🌱 RECOMMEND CROP",
                type="primary",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown("### 📊 Recommendation Results")
        
        if rec_submit:
            # Simulate crop recommendation logic
            if rec_temperature > 25 and rec_rainfall > 150 and rec_nitrogen > 80:
                recommended_crop = "Rice"
                confidence = 85
                reason = "High temperature, good rainfall, and sufficient nitrogen"
            elif rec_temperature < 22 and rec_rainfall < 120 and rec_phosphorus > 40:
                recommended_crop = "Wheat"
                confidence = 82
                reason = "Cool temperature, moderate rainfall, good phosphorus"
            elif rec_temperature < 20 and rec_ph < 7:
                recommended_crop = "Apple"
                confidence = 78
                reason = "Cool temperature, slightly acidic soil"
            elif rec_temperature > 28 and rec_potassium > 50:
                recommended_crop = "Cotton"
                confidence = 75
                reason = "Hot climate, good potassium levels"
            else:
                recommended_crop = "Maize"
                confidence = 70
                reason = "Adaptable to various conditions"
            
            # Get probabilities for top crops
            crops_rec = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Apple']
            probs = np.random.dirichlet(np.ones(5), size=1)[0] * 100
            probs = np.round(probs, 1)
            
            # Display results
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("## 🏆 RECOMMENDED CROP")
            st.markdown(f"# {recommended_crop.upper()}")
            st.markdown(f"**Confidence:** {confidence}%")
            st.markdown(f"**Reason:** {reason}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence gauge
            fig_confidence = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Recommendation Confidence"},
                gauge={'axis': {'range': [None, 100]}}
            ))
            fig_confidence.update_layout(height=250)
            st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Top crops chart
            fig_crops = px.bar(
                x=crops_rec,
                y=probs,
                color=probs,
                title="All Crop Suitability Scores",
                labels={'x': 'Crop', 'y': 'Suitability (%)'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_crops, use_container_width=True)
            
            # Crop requirements
            st.markdown("#### 📋 Crop Requirements")
            
            requirements = {
                'Rice': "🌡️ 22-32°C | 💧 1000-2500 mm | 🏜️ Clayey soil | 🧪 High N requirement",
                'Wheat': "🌡️ 15-25°C | 💧 300-600 mm | 🏜️ Loamy soil | 🧪 Balanced NPK",
                'Maize': "🌡️ 18-27°C | 💧 500-800 mm | 🏜️ Fertile soil | 🧪 High N requirement",
                'Cotton': "🌡️ 25-35°C | 💧 500-800 mm | 🏜️ Black soil | 🧪 Moderate NPK",
                'Apple': "🌡️ 10-25°C | 💧 800-1200 mm | 🏜️ Well-drained | 🧪 Moderate nutrients"
            }
            
            st.info(f"**{recommended_crop} Requirements:**\n\n{requirements.get(recommended_crop, 'Check detailed crop guide')}")

with pred_tabs[2]:  # Compare Crops
    st.markdown("## 📊 Crop Comparison")
    st.markdown("### Compare multiple crops for your conditions")
    
    # Select crops to compare
    selected_crops = st.multiselect(
        "Select crops to compare:",
        crops,
        default=['Rice', 'Wheat', 'Maize'],
        max_selections=5
    )
    
    if selected_crops:
        # Create comparison data
        comparison_data = []
        for crop in selected_crops:
            if crop in crop_info:
                info = crop_info[crop]
                # Simulate yield based on average conditions
                if crop == 'Rice':
                    yield_est = 4500
                elif crop == 'Wheat':
                    yield_est = 3500
                elif crop == 'Apple':
                    yield_est = 25000
                else:
                    yield_est = 4000
                
                comparison_data.append({
                    'Crop': crop,
                    'Avg Yield (kg/ha)': yield_est,
                    'Temp Range (°C)': f"{info['temp_range'][0]}-{info['temp_range'][1]}",
                    'Rain Range (mm)': f"{info['rain_range'][0]}-{info['rain_range'][1]}",
                    'pH Range': f"{info['ph_range'][0]}-{info['ph_range'][1]}",
                    'Water Need': 'High' if crop in ['Rice', 'Sugarcane'] else 'Medium',
                    'Season': 'Kharif' if crop in ['Rice', 'Cotton'] else 'Rabi' if crop == 'Wheat' else 'Both'
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(
            df_comparison.style.format({
                'Avg Yield (kg/ha)': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Visual comparison
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            # Yield comparison chart
            fig_yield = px.bar(
                df_comparison,
                x='Crop',
                y='Avg Yield (kg/ha)',
                color='Avg Yield (kg/ha)',
                title='Yield Comparison',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_yield, use_container_width=True)
        
        with col_vis2:
            # Radar chart for requirements
            categories = ['Yield Potential', 'Water Need', 'Temp Tolerance', 'Soil Adaptability']
            
            fig_radar = go.Figure()
            
            for crop in selected_crops:
                # Simulate scores (replace with actual logic)
                if crop == 'Rice':
                    values = [9, 3, 7, 6]  # High water need = lower score
                elif crop == 'Wheat':
                    values = [7, 8, 8, 9]
                elif crop == 'Maize':
                    values = [8, 7, 8, 8]
                else:
                    values = [6, 7, 6, 7]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=crop
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=True,
                title="Crop Suitability Radar"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Recommendation based on comparison
        st.markdown("#### 🎯 Comparison Insights")
        
        if len(selected_crops) >= 2:
            insights = []
            if 'Rice' in selected_crops and 'Wheat' in selected_crops:
                insights.append("🌾 **Rice vs Wheat**: Rice needs more water but gives higher yield in suitable conditions")
            
            if 'Apple' in selected_crops:
                insights.append("🍎 **Apple**: Higher investment but good returns, needs cooler climate")
            
            if 'Cotton' in selected_crops:
                insights.append("🧵 **Cotton**: Commercial crop with stable market prices")
            
            for insight in insights:
                st.info(insight)

# Footer
st.markdown("---")
st.markdown("🌾 **Smart Yield Predictor** | Using Advanced ML Algorithms | 📊 **Real-time Analysis**")