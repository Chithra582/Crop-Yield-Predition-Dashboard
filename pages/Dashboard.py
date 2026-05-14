# pages/1_🏠_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crop AI Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #2E7D32, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 20px;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .weather-card {
        background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data
@st.cache_data
def load_sample_data():
    """Load or generate sample data for dashboard"""
    np.random.seed(42)
    
    # Generate recent predictions
    predictions = pd.DataFrame({
        'Date': pd.date_range('2024-01-20', periods=15, freq='D'),
        'Crop': np.random.choice(['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'], 15),
        'Location': np.random.choice(['Punjab', 'Maharashtra', 'UP', 'Karnataka', 'West Bengal'], 15),
        'Yield': np.random.uniform(2000, 6000, 15),
        'Confidence': np.random.uniform(85, 98, 15)
    })
    
    # Generate crop yield data
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Potato', 'Tomato', 'Apple']
    crop_yield = pd.DataFrame({
        'Crop': crops,
        'Avg_Yield': [4500, 3500, 4000, 1800, 70000, 25000, 30000, 25000],
        'Optimal_Temp': [28, 20, 23, 30, 25, 20, 22, 18],
        'Water_Need': ['High', 'Medium', 'Medium', 'Medium', 'High', 'High', 'Medium', 'Medium'],
        'Season': ['Kharif', 'Rabi', 'Both', 'Kharif', 'Year-round', 'Rabi', 'Year-round', 'Year-round']
    })
    
    # Generate weather data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    weather = pd.DataFrame({
        'Date': dates,
        'Temperature': 25 + 8 * np.sin(2 * np.pi * np.arange(30) / 30) + np.random.normal(0, 2, 30),
        'Rainfall': np.maximum(0, np.random.poisson(3, 30).cumsum() / 10 + 5 * np.sin(2 * np.pi * np.arange(30) / 30)),
        'Humidity': 65 + 15 * np.sin(2 * np.pi * np.arange(30) / 30 + np.pi/3) + np.random.normal(0, 5, 30),
        'Soil_Moisture': 50 + 20 * np.sin(2 * np.pi * np.arange(30) / 30 + np.pi/6) + np.random.normal(0, 3, 30)
    })
    
    return predictions, crop_yield, weather

# Load data
predictions, crop_yield, weather = load_sample_data()

# Main Dashboard Content
st.markdown('<h1 class="main-header">🌾 SMART CROP AI DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown("### Complete Agricultural Intelligence Platform")

# Top Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🌾 Total Crops", "8", "+2 new")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📈 Predictions Today", "24", "+5")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🎯 ML Accuracy", "94.2%", "+1.8%")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("⏱️ Avg Response Time", "0.8s", "-0.2s")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Dashboard Sections
section_tabs = st.tabs(["📊 Overview", "📈 Analytics", "🌤️ Weather", "🚀 Quick Actions"])

with section_tabs[0]:  # Overview
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📈 Recent Yield Predictions")
        
        # Create interactive chart
        fig = px.line(
            predictions,
            x='Date',
            y='Yield',
            color='Crop',
            markers=True,
            title='Recent Yield Predictions Trend',
            labels={'Yield': 'Predicted Yield (kg/ha)', 'Date': 'Prediction Date'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions table
        st.markdown("#### 📋 Recent Predictions")
        recent_data = predictions.sort_values('Date', ascending=False).head(5)
        st.dataframe(
            recent_data.style.format({
                'Yield': '{:,.0f} kg/ha',
                'Confidence': '{:.1f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with col_right:
        st.markdown("### 🌡️ Current Conditions")
        
        # Weather card
        current_weather = weather.iloc[-1]
        st.markdown('<div class="weather-card">', unsafe_allow_html=True)
        st.markdown("#### Current Weather")
        col_temp, col_humid = st.columns(2)
        with col_temp:
            st.metric("🌡️ Temperature", f"{current_weather['Temperature']:.1f}°C")
        with col_humid:
            st.metric("💧 Humidity", f"{current_weather['Humidity']:.1f}%")
        st.markdown(f"**🌧️ Rainfall:** {current_weather['Rainfall']:.1f} mm")
        st.markdown(f"**🌱 Soil Moisture:** {current_weather['Soil_Moisture']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Crop status
        st.markdown("#### 🌾 Crop Status")
        for crop in ['Rice', 'Wheat', 'Maize'][:3]:
            crop_data = crop_yield[crop_yield['Crop'] == crop].iloc[0]
            with st.expander(f"{crop} - {crop_data['Season']}"):
                st.write(f"**Avg Yield:** {crop_data['Avg_Yield']:,.0f} kg/ha")
                st.write(f"**Optimal Temp:** {crop_data['Optimal_Temp']}°C")
                st.write(f"**Water Need:** {crop_data['Water_Need']}")
        
        # System status
        st.markdown("#### ⚙️ System Status")
        st.success("✅ All systems operational")
        st.info("📡 Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

with section_tabs[1]:  # Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌾 Crop Yield Distribution")
        
        # Bar chart of average yields
        fig1 = px.bar(
            crop_yield,
            x='Crop',
            y='Avg_Yield',
            color='Avg_Yield',
            title='Average Yield by Crop',
            labels={'Avg_Yield': 'Yield (kg/ha)', 'Crop': 'Crop Type'},
            color_continuous_scale='Viridis'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Crop statistics
        st.markdown("#### 📊 Crop Statistics")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric("Highest Yield", f"{crop_yield['Avg_Yield'].max():,.0f} kg/ha", "Sugarcane")
        with stats_col2:
            st.metric("Lowest Yield", f"{crop_yield['Avg_Yield'].min():,.0f} kg/ha", "Cotton")
    
    with col2:
        st.markdown("### 📈 Prediction Analytics")
        
        # Confidence distribution
        fig2 = px.histogram(
            predictions,
            x='Confidence',
            nbins=20,
            title='Prediction Confidence Distribution',
            labels={'Confidence': 'Confidence Level (%)'},
            color_discrete_sequence=['#4CAF50']
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Top performing predictions
        st.markdown("#### 🏆 Top Predictions")
        top_predictions = predictions.nlargest(3, 'Confidence')
        for idx, row in top_predictions.iterrows():
            st.info(f"**{row['Crop']}** at {row['Location']}: {row['Yield']:,.0f} kg/ha ({row['Confidence']:.1f}% confidence)")

with section_tabs[2]:  # Weather
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌤️ Weather Trends")
        
        # Temperature trend
        fig_temp = px.line(
            weather,
            x='Date',
            y='Temperature',
            title='Temperature Trend (Last 30 Days)',
            labels={'Temperature': 'Temperature (°C)', 'Date': 'Date'}
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Rainfall trend
        fig_rain = px.area(
            weather,
            x='Date',
            y='Rainfall',
            title='Rainfall Trend (Last 30 Days)',
            labels={'Rainfall': 'Rainfall (mm)', 'Date': 'Date'}
        )
        st.plotly_chart(fig_rain, use_container_width=True)
    
    with col2:
        st.markdown("### 🌧️ Weather Analysis")
        
        # Current conditions gauge
        current_temp = weather.iloc[-1]['Temperature']
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_temp,
            title={'text': "Current Temperature"},
            gauge={
                'axis': {'range': [0, 40]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 15], 'color': "lightblue"},
                    {'range': [15, 25], 'color': "lightgreen"},
                    {'range': [25, 35], 'color': "yellow"},
                    {'range': [35, 40], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Weather statistics
        st.markdown("#### 📊 Weather Stats")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Avg Temperature", f"{weather['Temperature'].mean():.1f}°C")
            st.metric("Max Temperature", f"{weather['Temperature'].max():.1f}°C")
        with col_stat2:
            st.metric("Total Rainfall", f"{weather['Rainfall'].sum():.1f} mm")
            st.metric("Avg Humidity", f"{weather['Humidity'].mean():.1f}%")
        
        # Weather forecast
        st.markdown("#### 🌤️ 7-Day Forecast")
        forecast_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        forecast_temp = [25, 26, 24, 27, 28, 26, 25]
        forecast_rain = [0, 5, 10, 0, 0, 2, 0]
        
        for day, temp, rain in zip(forecast_days, forecast_temp, forecast_rain):
            col_day, col_temp, col_rain = st.columns([1, 2, 2])
            with col_day:
                st.write(f"**{day}**")
            with col_temp:
                st.write(f"🌡️ {temp}°C")
            with col_rain:
                st.write(f"🌧️ {rain} mm")

with section_tabs[3]:  # Quick Actions
    st.markdown("## 🚀 Quick Actions")
    
    # Quick action cards in grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Predict Yield")
        st.markdown("Get instant yield prediction for your crops using AI")
        if st.button("Go to Predictor", use_container_width=True):
            st.switch_page("pages/2_📈_Predictor.py")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ View Maps")
        st.markdown("Interactive maps for crop suitability and soil analysis")
        if st.button("Explore Maps", use_container_width=True):
            st.switch_page("pages/3_🗺️_Live_Maps.py")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🎤 Voice Assistant")
        st.markdown("Voice-controlled commands for hands-free operation")
        if st.button("Start Voice Assistant", use_container_width=True):
            st.switch_page("pages/4_🎤_Voice_Assistant.py")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Chatbot")
        st.markdown("24/7 agricultural advisory and Q&A support")
        if st.button("Chat Now", use_container_width=True):
            st.switch_page("pages/5_🤖_Crop_Chatbot.py")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Analysis")
        st.markdown("Comprehensive analytics and visualizations")
        if st.button("View Analytics", use_container_width=True):
            st.info("Analytics loaded in current view")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Reports")
        st.markdown("Generate detailed agricultural reports")
        if st.button("Create Report", use_container_width=True):
            st.success("Report generation started...")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("---")
    st.markdown("### 📝 Recent Activity")
    
    activities = [
        {"time": "10:30 AM", "action": "Yield prediction completed", "crop": "Rice", "user": "Farmer John"},
        {"time": "09:45 AM", "action": "Soil analysis report generated", "location": "Punjab", "user": "Agri Expert"},
        {"time": "Yesterday", "action": "New crop data uploaded", "crops": "5 new varieties", "user": "Admin"},
        {"time": "Jan 26", "action": "ML model retrained", "accuracy": "94.2%", "user": "System"}
    ]
    
    for activity in activities:
        col_time, col_action, col_details = st.columns([1, 2, 2])
        with col_time:
            st.write(f"🕒 {activity['time']}")
        with col_action:
            st.write(f"**{activity['action']}**")
        with col_details:
            details = [f"{k}: {v}" for k, v in activity.items() if k not in ['time', 'action']]
            st.write(", ".join(details))

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col1:
    st.markdown("🌾 **Crop AI**")

with footer_col2:
    st.markdown("<center>🚀 Real-time Agricultural Intelligence | Powered by Machine Learning</center>", unsafe_allow_html=True)

with footer_col3:
    st.markdown("📱 **v3.1**")

st.markdown("<center><small>© 2024 Smart Agriculture Dashboard | Data updates every 15 minutes</small></center>", unsafe_allow_html=True)