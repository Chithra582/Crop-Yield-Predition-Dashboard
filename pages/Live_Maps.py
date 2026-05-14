# pages/3_🗺️_Live_Maps.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Live Maps",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .map-header {
        background: linear-gradient(90deg, #2196F3, #03A9F4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .map-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
    }
    
    .location-card {
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .legend-item {
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 8px;
        border-radius: 3px;
    }
    
    .stSelectbox > div > div {
        border-color: #2196F3 !important;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample map data
@st.cache_data
def generate_map_data():
    """Generate sample data for maps"""
    # Indian states data
    states = ['Punjab', 'Maharashtra', 'Karnataka', 'Uttar Pradesh', 
              'West Bengal', 'Tamil Nadu', 'Rajasthan', 'Gujarat',
              'Madhya Pradesh', 'Bihar', 'Odisha', 'Andhra Pradesh']
    
    # Coordinates (simplified)
    state_coords = {
        'Punjab': [31.1471, 75.3412],
        'Maharashtra': [19.7515, 75.7139],
        'Karnataka': [15.3173, 75.7139],
        'Uttar Pradesh': [26.8467, 80.9462],
        'West Bengal': [22.9868, 87.8550],
        'Tamil Nadu': [11.1271, 78.6569],
        'Rajasthan': [27.0238, 74.2179],
        'Gujarat': [22.2587, 71.1924],
        'Madhya Pradesh': [23.2599, 77.4126],
        'Bihar': [25.0961, 85.3131],
        'Odisha': [20.9517, 85.0985],
        'Andhra Pradesh': [15.9129, 79.7400]
    }
    
    # Crop suitability scores
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane']
    
    suitability_data = []
    for state in states:
        for crop in crops:
            # Generate realistic suitability scores
            if crop == 'Rice':
                score = 90 if state in ['West Bengal', 'Andhra Pradesh', 'Odisha'] else np.random.randint(60, 85)
            elif crop == 'Wheat':
                score = 92 if state in ['Punjab', 'Uttar Pradesh'] else np.random.randint(65, 88)
            elif crop == 'Cotton':
                score = 88 if state in ['Maharashtra', 'Gujarat'] else np.random.randint(60, 82)
            else:
                score = np.random.randint(70, 90)
            
            suitability_data.append({
                'State': state,
                'Crop': crop,
                'Suitability': score,
                'Lat': state_coords[state][0],
                'Lon': state_coords[state][1]
            })
    
    df_suitability = pd.DataFrame(suitability_data)
    
    # Soil quality data
    soil_data = []
    for state in states:
        soil_data.append({
            'State': state,
            'Soil pH': np.random.uniform(6.0, 7.5),
            'Nitrogen': np.random.randint(60, 120),
            'Phosphorus': np.random.randint(30, 70),
            'Potassium': np.random.randint(40, 90),
            'Organic Matter': np.random.uniform(0.5, 2.5),
            'Soil Type': np.random.choice(['Loamy', 'Clayey', 'Sandy', 'Silty'])
        })
    
    df_soil = pd.DataFrame(soil_data)
    
    # Weather data
    weather_data = []
    for state in states:
        weather_data.append({
            'State': state,
            'Temperature': np.random.uniform(20, 32),
            'Rainfall': np.random.uniform(500, 1500),
            'Humidity': np.random.uniform(60, 85),
            'Growing Days': np.random.randint(180, 300)
        })
    
    df_weather = pd.DataFrame(weather_data)
    
    return df_suitability, df_soil, df_weather, state_coords

# Load data
df_suitability, df_soil, df_weather, state_coords = generate_map_data()

# Main Maps Content
st.markdown('<h1 class="map-header">🗺️ LIVE CROP SUITABILITY MAPS</h1>', unsafe_allow_html=True)
st.markdown("### Interactive Maps for Agricultural Planning")

# Create tabs for different map types
map_tabs = st.tabs(["🌍 Crop Suitability", "🧪 Soil Quality", "🌤️ Weather Maps", "📊 Region Analysis"])

with map_tabs[0]:  # Crop Suitability
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        st.markdown("### 🌾 Crop Suitability by Region")
        
        # Crop selection
        selected_crop = st.selectbox(
            "Select Crop for Suitability Map:",
            ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'],
            key="map_crop"
        )
        
        # Filter data for selected crop
        crop_data = df_suitability[df_suitability['Crop'] == selected_crop]
        
        # Create choropleth-like visualization
        fig = px.scatter_geo(
            crop_data,
            lat='Lat',
            lon='Lon',
            size='Suitability',
            color='Suitability',
            hover_name='State',
            hover_data={'Suitability': ':.1f', 'Lat': False, 'Lon': False},
            title=f'{selected_crop} Suitability by State',
            color_continuous_scale='Viridis',
            size_max=30,
            projection='natural earth'
        )
        
        # Update layout for India focus
        fig.update_geos(
            visible=False,
            resolution=50,
            showcountries=True,
            countrycolor="Black",
            showsubunits=True,
            subunitcolor="Blue"
        )
        
        fig.update_layout(
            geo=dict(
                lonaxis_range=[68, 98],  # India longitude range
                lataxis_range=[8, 37],   # India latitude range
                projection_scale=5
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Suitability details
        st.markdown("#### 📊 Suitability Details")
        
        # Top 5 states for selected crop
        top_states = crop_data.nlargest(5, 'Suitability')
        for idx, row in top_states.iterrows():
            with st.expander(f"{row['State']} - {row['Suitability']:.1f}% suitable"):
                st.write(f"**Best for {selected_crop} cultivation**")
                st.write(f"**Soil Type:** {df_soil[df_soil['State'] == row['State']]['Soil Type'].iloc[0]}")
                st.write(f"**Avg Temperature:** {df_weather[df_weather['State'] == row['State']]['Temperature'].iloc[0]:.1f}°C")
                st.write(f"**Annual Rainfall:** {df_weather[df_weather['State'] == row['State']]['Rainfall'].iloc[0]:.0f} mm")
    
    with col2:
        st.markdown("### 🏆 Top Regions")
        
        # Top 5 states for selected crop
        for idx, row in top_states.iterrows():
            st.markdown(f'<div class="location-card">', unsafe_allow_html=True)
            st.markdown(f"#### #{idx+1} {row['State']}")
            st.markdown(f"**Suitability:** {row['Suitability']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Legend
        st.markdown("#### 🎨 Suitability Legend")
        st.markdown('<div class="legend-item" style="background-color: #440154;"></div> <span>Low (60-70%)</span><br>', unsafe_allow_html=True)
        st.markdown('<div class="legend-item" style="background-color: #31688e;"></div> <span>Medium (70-80%)</span><br>', unsafe_allow_html=True)
        st.markdown('<div class="legend-item" style="background-color: #35b779;"></div> <span>High (80-90%)</span><br>', unsafe_allow_html=True)
        st.markdown('<div class="legend-item" style="background-color: #fde725;"></div> <span>Very High (90-100%)</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("#### 📈 Quick Stats")
        st.metric("Avg Suitability", f"{crop_data['Suitability'].mean():.1f}%")
        st.metric("Best State", top_states.iloc[0]['State'])
        st.metric("Worst State", crop_data.nsmallest(1, 'Suitability')['State'].iloc[0])

with map_tabs[1]:  # Soil Quality
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        st.markdown("### 🧪 Soil Quality Map")
        
        # Soil parameter selection
        soil_param = st.selectbox(
            "Select Soil Parameter:",
            ['Soil pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic Matter'],
            key="soil_param"
        )
        
        # Create bar chart for soil parameter
        fig_soil = px.bar(
            df_soil.sort_values(soil_param, ascending=False),
            x='State',
            y=soil_param,
            color=soil_param,
            title=f'{soil_param} Distribution by State',
            labels={soil_param: soil_param},
            color_continuous_scale='RdBu' if soil_param == 'Soil pH' else 'Viridis'
        )
        fig_soil.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig_soil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        st.markdown("### 🌱 Soil Type Distribution")
        
        # Soil type pie chart
        soil_type_counts = df_soil['Soil Type'].value_counts().reset_index()
        soil_type_counts.columns = ['Soil Type', 'Count']
        
        fig_pie = px.pie(
            soil_type_counts,
            values='Count',
            names='Soil Type',
            title='Soil Type Distribution Across States',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Soil quality table
    st.markdown("#### 📋 Detailed Soil Analysis")
    st.dataframe(
        df_soil.style.format({
            'Soil pH': '{:.2f}',
            'Organic Matter': '{:.2f}%'
        }).background_gradient(subset=['Nitrogen', 'Phosphorus', 'Potassium'], cmap='YlGn'),
        use_container_width=True,
        height=300
    )

with map_tabs[2]:  # Weather Maps
    st.markdown("### 🌤️ Agricultural Weather Maps")
    
    col_weather1, col_weather2 = st.columns(2)
    
    with col_weather1:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        st.markdown("#### 🌡️ Temperature Map")
        
        fig_temp = px.choropleth(
            df_weather,
            locations='State',
            locationmode='country names',
            color='Temperature',
            hover_name='State',
            title='Average Temperature by State (°C)',
            color_continuous_scale='RdYlBu_r'
        )
        fig_temp.update_layout(height=400)
        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_weather2:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        st.markdown("#### 🌧️ Rainfall Map")
        
        fig_rain = px.choropleth(
            df_weather,
            locations='State',
            locationmode='country names',
            color='Rainfall',
            hover_name='State',
            title='Annual Rainfall by State (mm)',
            color_continuous_scale='Blues'
        )
        fig_rain.update_layout(height=400)
        st.plotly_chart(fig_rain, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Growing season analysis
    st.markdown("#### 📅 Growing Season Analysis")
    
    col_grow1, col_grow2 = st.columns(2)
    
    with col_grow1:
        # Growing days bar chart
        fig_grow = px.bar(
            df_weather.sort_values('Growing Days', ascending=False),
            x='State',
            y='Growing Days',
            color='Growing Days',
            title='Growing Days per Year by State',
            color_continuous_scale='Greens'
        )
        fig_grow.update_layout(height=300, xaxis_tickangle=45)
        st.plotly_chart(fig_grow, use_container_width=True)
    
    with col_grow2:
        # Humidity heatmap
        fig_humid = px.density_heatmap(
            df_weather,
            x='State',
            y='Temperature',
            z='Humidity',
            title='Temperature-Humidity Relationship',
            color_continuous_scale='Viridis'
        )
        fig_humid.update_layout(height=300, xaxis_tickangle=45)
        st.plotly_chart(fig_humid, use_container_width=True)

with map_tabs[3]:  # Region Analysis
    st.markdown("### 📊 Region-Specific Analysis")
    
    # Select region for detailed analysis
    selected_region = st.selectbox(
        "Select Region for Detailed Analysis:",
        df_suitability['State'].unique()
    )
    
    if selected_region:
        col_region1, col_region2 = st.columns([1, 1])
        
        with col_region1:
            st.markdown(f'<div class="map-card">', unsafe_allow_html=True)
            st.markdown(f"#### 🌾 Crop Suitability for {selected_region}")
            
            # Get crop suitability for selected region
            region_crops = df_suitability[df_suitability['State'] == selected_region]
            
            fig_region = px.bar(
                region_crops.sort_values('Suitability', ascending=True),
                x='Suitability',
                y='Crop',
                orientation='h',
                color='Suitability',
                title=f'Crop Suitability in {selected_region}',
                color_continuous_scale='Viridis'
            )
            fig_region.update_layout(height=400)
            st.plotly_chart(fig_region, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_region2:
            st.markdown(f'<div class="map-card">', unsafe_allow_html=True)
            st.markdown(f"#### 📈 Region Statistics")
            
            # Get region data
            region_soil = df_soil[df_soil['State'] == selected_region].iloc[0]
            region_weather = df_weather[df_weather['State'] == selected_region].iloc[0]
            
            # Display metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("🌡️ Temperature", f"{region_weather['Temperature']:.1f}°C")
                st.metric("💧 Humidity", f"{region_weather['Humidity']:.1f}%")
                st.metric("🧪 Soil pH", f"{region_soil['Soil pH']:.2f}")
            
            with col_metric2:
                st.metric("🌧️ Rainfall", f"{region_weather['Rainfall']:.0f} mm")
                st.metric("📅 Growing Days", f"{region_weather['Growing Days']}")
                st.metric("🏜️ Soil Type", region_soil['Soil Type'])
            
            # Nutrient levels gauge
            st.markdown("##### 🧪 Nutrient Levels")
            
            fig_nutrients = go.Figure()
            
            nutrients = ['Nitrogen', 'Phosphorus', 'Potassium']
            values = [region_soil['Nitrogen'], region_soil['Phosphorus'], region_soil['Potassium']]
            optimal_ranges = [(80, 120), (40, 80), (50, 100)]
            
            for i, (nutrient, value, (opt_min, opt_max)) in enumerate(zip(nutrients, values, optimal_ranges)):
                fig_nutrients.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': nutrient},
                    gauge={
                        'axis': {'range': [0, 150]},
                        'bar': {'color': "#4CAF50" if opt_min <= value <= opt_max else "#FF9800"},
                        'steps': [
                            {'range': [0, opt_min], 'color': "lightgray"},
                            {'range': [opt_min, opt_max], 'color': "lightgreen"},
                            {'range': [opt_max, 150], 'color': "lightcoral"}
                        ]
                    },
                    domain={'row': 0, 'column': i}
                ))
            
            fig_nutrients.update_layout(
                grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
                height=200
            )
            
            st.plotly_chart(fig_nutrients, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations for selected region
        st.markdown("#### 💡 Recommendations for Region")
        
        # Get best crop for region
        best_crop = region_crops.nlargest(1, 'Suitability').iloc[0]
        
        recommendations = []
        
        if region_soil['Nitrogen'] < 80:
            recommendations.append(f"**Add nitrogen fertilizer** - Current N level ({region_soil['Nitrogen']} kg/ha) is below optimal")
        
        if region_weather['Rainfall'] < 500:
            recommendations.append(f"**Implement irrigation system** - Low rainfall ({region_weather['Rainfall']:.0f} mm) requires supplemental water")
        
        if region_soil['Soil pH'] < 6.0:
            recommendations.append(f"**Apply lime** - Soil is acidic (pH: {region_soil['Soil pH']:.2f})")
        
        recommendations.append(f"**Consider {best_crop['Crop']} cultivation** - Highest suitability ({best_crop['Suitability']:.1f}%) for this region")
        
        for rec in recommendations:
            st.info(rec)

# Footer
st.markdown("---")
st.markdown("🗺️ **Interactive Agricultural Maps** | Real-time Data | 📍 **Location-based Insights**")