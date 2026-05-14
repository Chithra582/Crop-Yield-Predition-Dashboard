import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')

class MapGenerator:
    """
    Generates interactive maps and visualizations for agricultural data.
    """
    
    def __init__(self, data_loader=None):
        """
        Initialize MapGenerator.
        
        Args:
            data_loader: DataLoader instance (optional)
        """
        self.data_loader = data_loader
        
        # Default map settings for India
        self.default_center = [20.5937, 78.9629]  # Center of India
        self.default_zoom = 5
        self.tile_layers = {
            'OpenStreetMap': 'OpenStreetMap',
            'CartoDB Positron': 'CartoDB Positron',
            'CartoDB Dark Matter': 'CartoDB Dark_Matter',
            'Stamen Terrain': 'Stamen Terrain',
            'Stamen Toner': 'Stamen Toner'
        }
    
    def create_yield_heatmap(self, yield_data: pd.DataFrame, 
                            crop_filter: str = None,
                            state_filter: str = None,
                            season_filter: str = None) -> folium.Map:
        """
        Create a heatmap of crop yields.
        
        Args:
            yield_data: DataFrame containing yield data
            crop_filter: Filter by crop type (optional)
            state_filter: Filter by state (optional)
            season_filter: Filter by season (optional)
            
        Returns:
            folium.Map object with heatmap
        """
        # Apply filters
        filtered_data = yield_data.copy()
        
        if crop_filter:
            filtered_data = filtered_data[filtered_data['Crop'] == crop_filter]
        
        if state_filter:
            filtered_data = filtered_data[filtered_data['State'] == state_filter]
        
        if season_filter:
            filtered_data = filtered_data[filtered_data['Season'] == season_filter]
        
        if filtered_data.empty:
            # Create empty map with message
            m = folium.Map(location=self.default_center, zoom_start=self.default_zoom)
            folium.Marker(
                self.default_center,
                popup="No data available for selected filters",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            return m
        
        # Create base map
        m = folium.Map(location=self.default_center, zoom_start=self.default_zoom)
        
        # Add tile layer selector
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB Dark_Matter').add_to(m)
        folium.LayerControl().add_to(m)
        
        # Prepare heatmap data
        heat_data = []
        
        # If we have latitude/longitude data, use it
        if 'Latitude' in filtered_data.columns and 'Longitude' in filtered_data.columns:
            for _, row in filtered_data.iterrows():
                if not pd.isna(row['Latitude']) and not pd.isna(row['Longitude']):
                    # Weight by yield
                    weight = min(row['Yield'] / 1000, 10)  # Scale for visibility
                    heat_data.append([row['Latitude'], row['Longitude'], weight])
        else:
            # Generate approximate coordinates based on state/district
            for _, row in filtered_data.iterrows():
                lat, lon = self._get_approximate_coordinates(row.get('State'), row.get('District'))
                if lat and lon:
                    weight = min(row['Yield'] / 1000, 10)
                    heat_data.append([lat, lon, weight])
        
        if heat_data:
            # Add heatmap
            plugins.HeatMap(
                heat_data,
                radius=15,
                blur=10,
                max_zoom=1,
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
            ).add_to(m)
        
        # Add markers for top yield locations
        top_yields = filtered_data.nlargest(10, 'Yield')
        for _, row in top_yields.iterrows():
            lat, lon = self._get_approximate_coordinates(row.get('State'), row.get('District'))
            if lat and lon:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=min(row['Yield'] / 500, 20),
                    popup=self._create_yield_popup(row),
                    color='green',
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
        
        # Add title
        title_html = '''
            <h3 align="center" style="font-size:16px"><b>Crop Yield Heatmap</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_soil_quality_map(self, soil_data: pd.DataFrame, 
                               soil_parameter: str = 'pH') -> folium.Map:
        """
        Create a map showing soil quality parameters.
        
        Args:
            soil_data: DataFrame containing soil data
            soil_parameter: Soil parameter to visualize (pH, Organic_Carbon, etc.)
            
        Returns:
            folium.Map object with soil quality visualization
        """
        if soil_data.empty:
            m = folium.Map(location=self.default_center, zoom_start=self.default_zoom)
            return m
        
        # Create base map
        m = folium.Map(location=self.default_center, zoom_start=self.default_zoom)
        
        # Color scale based on parameter
        if soil_parameter == 'pH':
            # pH scale: acidic (red) to alkaline (blue)
            def get_color(val):
                if val < 5.5:
                    return 'red'
                elif val < 6.5:
                    return 'orange'
                elif val < 7.5:
                    return 'green'
                else:
                    return 'blue'
            param_name = 'pH Level'
        elif soil_parameter == 'Organic_Carbon':
            # Organic carbon: low (red) to high (green)
            def get_color(val):
                if val < 0.8:
                    return 'red'
                elif val < 1.5:
                    return 'orange'
                elif val < 2.5:
                    return 'yellow'
                else:
                    return 'green'
            param_name = 'Organic Carbon (%)'
        else:
            # Default color scale
            def get_color(val):
                if val < soil_data[soil_parameter].quantile(0.25):
                    return 'red'
                elif val < soil_data[soil_parameter].quantile(0.5):
                    return 'orange'
                elif val < soil_data[soil_parameter].quantile(0.75):
                    return 'yellow'
                else:
                    return 'green'
            param_name = soil_parameter
        
        # Add soil data points
        for _, row in soil_data.iterrows():
            # Generate random coordinates for demo (in real app, use actual coordinates)
            lat = np.random.uniform(8, 37)
            lon = np.random.uniform(68, 97)
            
            value = row[soil_parameter] if soil_parameter in row else 0
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=self._create_soil_popup(row, soil_parameter, value),
                color=get_color(value),
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        
        # Add legend
        self._add_soil_legend(m, soil_parameter)
        
        # Add title
        title_html = f'''
            <h3 align="center" style="font-size:16px"><b>Soil {param_name} Distribution</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_crop_distribution_map(self, recommendation_data: pd.DataFrame,
                                    crop_name: str = None) -> folium.Map:
        """
        Create a map showing crop distribution.
        
        Args:
            recommendation_data: DataFrame containing crop recommendation data
            crop_name: Specific crop to visualize (optional)
            
        Returns:
            folium.Map object with crop distribution
        """
        if recommendation_data.empty:
            m = folium.Map(location=self.default_center, zoom_start=self.default_zoom)
            return m
        
        # Filter by crop if specified
        if crop_name:
            filtered_data = recommendation_data[recommendation_data['Crop'] == crop_name]
            if filtered_data.empty:
                # If no data for specific crop, use all data
                filtered_data = recommendation_data
                crop_name = "Multiple Crops"
        else:
            filtered_data = recommendation_data
            crop_name = "All Crops"
        
        # Get crop counts by state
        if 'State' in filtered_data.columns:
            crop_counts = filtered_data['State'].value_counts().reset_index()
            crop_counts.columns = ['State', 'Count']
        else:
            crop_counts = pd.DataFrame({'State': ['Unknown'], 'Count': [len(filtered_data)]})
        
        # Create base map
        m = folium.Map(location=self.default_center, zoom_start=self.default_zoom)
        
        # Add choropleth if we have state data
        if len(crop_counts) > 1:
            # Define colors for different counts
            max_count = crop_counts['Count'].max()
            
            for _, row in crop_counts.iterrows():
                state = row['State']
                count = row['Count']
                
                # Get approximate coordinates for state
                lat, lon = self._get_state_coordinates(state)
                if lat and lon:
                    # Calculate marker size based on count
                    radius = 5 + (count / max_count * 15)
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=f"<b>{state}</b><br>Crop Recommendations: {count}",
                        color='blue',
                        fill=True,
                        fill_opacity=0.6
                    ).add_to(m)
        
        # Add title
        title_html = f'''
            <h3 align="center" style="font-size:16px"><b>{crop_name} Distribution</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_interactive_plotly_map(self, data: pd.DataFrame,
                                     color_column: str = 'Yield',
                                     size_column: str = 'Area',
                                     hover_columns: List[str] = None) -> go.Figure:
        """
        Create an interactive Plotly scatter map.
        
        Args:
            data: DataFrame with data to plot
            color_column: Column to use for color coding
            size_column: Column to use for marker size
            hover_columns: Columns to show in hover info
            
        Returns:
            plotly.graph_objects.Figure object
        """
        if data.empty:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
        
        # Generate approximate coordinates if not present
        if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
            data = data.copy()
            data['Latitude'] = np.random.uniform(8, 37, len(data))
            data['Longitude'] = np.random.uniform(68, 97, len(data))
        
        # Set default hover columns
        if hover_columns is None:
            hover_columns = ['Crop', 'State', 'District', 'Season', 'Yield', 'Area']
            hover_columns = [col for col in hover_columns if col in data.columns]
        
        # Create scatter map
        fig = px.scatter_mapbox(
            data,
            lat='Latitude',
            lon='Longitude',
            color=color_column if color_column in data.columns else None,
            size=size_column if size_column in data.columns else None,
            hover_name='Crop' if 'Crop' in data.columns else None,
            hover_data=hover_columns,
            color_continuous_scale='Viridis',
            zoom=4,
            height=600,
            title=f"{color_column} Distribution Map"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            showlegend=True
        )
        
        return fig
    
    def create_yield_comparison_chart(self, yield_data: pd.DataFrame,
                                     crops: List[str] = None,
                                     states: List[str] = None) -> go.Figure:
        """
        Create a bar chart comparing yields.
        
        Args:
            yield_data: DataFrame containing yield data
            crops: List of crops to compare (optional)
            states: List of states to compare (optional)
            
        Returns:
            plotly.graph_objects.Figure object
        """
        filtered_data = yield_data.copy()
        
        # Apply filters
        if crops:
            filtered_data = filtered_data[filtered_data['Crop'].isin(crops)]
        
        if states:
            filtered_data = filtered_data[filtered_data['State'].isin(states)]
        
        if filtered_data.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available for selected filters")
            return fig
        
        # Group data for comparison
        if crops and not states:
            # Compare crops
            grouped = filtered_data.groupby('Crop')['Yield'].agg(['mean', 'std']).reset_index()
            grouped = grouped.sort_values('mean', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    name='Average Yield',
                    x=grouped['Crop'],
                    y=grouped['mean'],
                    error_y=dict(type='data', array=grouped['std'], visible=True),
                    marker_color='lightgreen'
                )
            ])
            
            fig.update_layout(
                title="Crop Yield Comparison",
                xaxis_title="Crop",
                yaxis_title="Yield (kg/ha)",
                showlegend=False
            )
            
        elif states and not crops:
            # Compare states
            grouped = filtered_data.groupby('State')['Yield'].agg(['mean', 'std']).reset_index()
            grouped = grouped.sort_values('mean', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    name='Average Yield',
                    x=grouped['State'],
                    y=grouped['mean'],
                    error_y=dict(type='data', array=grouped['std'], visible=True),
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="State-wise Yield Comparison",
                xaxis_title="State",
                yaxis_title="Yield (kg/ha)",
                showlegend=False
            )
        
        else:
            # Create grouped bar chart
            pivot_data = filtered_data.pivot_table(
                values='Yield',
                index='State' if states else 'Crop',
                columns='Crop' if crops else 'State',
                aggfunc='mean'
            ).fillna(0)
            
            fig = go.Figure()
            
            for column in pivot_data.columns:
                fig.add_trace(go.Bar(
                    name=column,
                    x=pivot_data.index,
                    y=pivot_data[column]
                ))
            
            fig.update_layout(
                title="Yield Comparison",
                xaxis_title="Category",
                yaxis_title="Yield (kg/ha)",
                barmode='group'
            )
        
        return fig
    
    def create_seasonal_yield_trend(self, yield_data: pd.DataFrame,
                                   crop: str = None,
                                   state: str = None) -> go.Figure:
        """
        Create a line chart showing seasonal yield trends.
        
        Args:
            yield_data: DataFrame containing yield data
            crop: Specific crop to analyze (optional)
            state: Specific state to analyze (optional)
            
        Returns:
            plotly.graph_objects.Figure object
        """
        filtered_data = yield_data.copy()
        
        # Apply filters
        if crop:
            filtered_data = filtered_data[filtered_data['Crop'] == crop]
        
        if state:
            filtered_data = filtered_data[filtered_data['State'] == state]
        
        if filtered_data.empty or 'Season' not in filtered_data.columns:
            fig = go.Figure()
            fig.update_layout(title="No seasonal data available")
            return fig
        
        # Group by season
        seasonal_data = filtered_data.groupby('Season')['Yield'].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        
        # Add average yield line
        fig.add_trace(go.Scatter(
            x=seasonal_data['Season'],
            y=seasonal_data['mean'],
            mode='lines+markers',
            name='Average Yield',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        
        # Add error bars
        fig.add_trace(go.Scatter(
            x=seasonal_data['Season'],
            y=seasonal_data['mean'] + seasonal_data['std'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=seasonal_data['Season'],
            y=seasonal_data['mean'] - seasonal_data['std'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(0, 100, 0, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
        
        title = "Seasonal Yield Trend"
        if crop:
            title += f" - {crop}"
        if state:
            title += f" in {state}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Season",
            yaxis_title="Yield (kg/ha)",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def _get_approximate_coordinates(self, state: str = None, district: str = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Get approximate coordinates for a state/district.
        
        Args:
            state: State name
            district: District name
            
        Returns:
            Tuple of (latitude, longitude) or (None, None)
        """
        # Indian states coordinates (approximate)
        state_coordinates = {
            'Punjab': (31.1471, 75.3412),
            'Haryana': (29.0588, 76.0856),
            'Uttar Pradesh': (26.8467, 80.9462),
            'Madhya Pradesh': (22.9734, 78.6569),
            'Rajasthan': (27.0238, 74.2179),
            'Maharashtra': (19.7515, 75.7139),
            'Karnataka': (15.3173, 75.7139),
            'Andhra Pradesh': (15.9129, 79.7400),
            'Tamil Nadu': (11.1271, 78.6569),
            'Gujarat': (22.2587, 71.1924),
            'West Bengal': (22.9868, 87.8550),
            'Bihar': (25.0961, 85.3131),
            'Odisha': (20.9517, 85.0985),
            'Kerala': (10.8505, 76.2711),
            'Assam': (26.2006, 92.9376),
            'Telangana': (17.1232, 79.2088)
        }
        
        if state and state in state_coordinates:
            lat, lon = state_coordinates[state]
            # Add small random offset for multiple points in same state
            lat += np.random.uniform(-1, 1)
            lon += np.random.uniform(-1, 1)
            return lat, lon
        
        # Default coordinates for India
        return self.default_center[0] + np.random.uniform(-5, 5), self.default_center[1] + np.random.uniform(-5, 5)
    
    def _get_state_coordinates(self, state: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get coordinates for a specific state.
        
        Args:
            state: State name
            
        Returns:
            Tuple of (latitude, longitude) or (None, None)
        """
        return self._get_approximate_coordinates(state)
    
    def _create_yield_popup(self, row: pd.Series) -> str:
        """
        Create HTML popup for yield data point.
        
        Args:
            row: DataFrame row with yield data
            
        Returns:
            HTML string for popup
        """
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Crop:</b> {row.get('Crop', 'N/A')}<br>
            <b>State:</b> {row.get('State', 'N/A')}<br>
            <b>District:</b> {row.get('District', 'N/A')}<br>
            <b>Season:</b> {row.get('Season', 'N/A')}<br>
            <b>Yield:</b> {row.get('Yield', 'N/A'):.1f} kg/ha<br>
            <b>Area:</b> {row.get('Area', 'N/A'):.1f} hectares<br>
        """
        
        if 'Rainfall' in row:
            popup_html += f"<b>Rainfall:</b> {row['Rainfall']:.1f} mm<br>"
        
        if 'Temperature' in row:
            popup_html += f"<b>Temperature:</b> {row['Temperature']:.1f}°C<br>"
        
        popup_html += "</div>"
        return popup_html
    
    def _create_soil_popup(self, row: pd.Series, parameter: str, value: float) -> str:
        """
        Create HTML popup for soil data point.
        
        Args:
            row: DataFrame row with soil data
            parameter: Soil parameter being visualized
            value: Parameter value
            
        Returns:
            HTML string for popup
        """
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Soil Type:</b> {row.get('Soil_Type', 'N/A')}<br>
            <b>{parameter}:</b> {value:.2f}<br>
        """
        
        # Add other soil parameters if available
        soil_params = ['pH', 'Organic_Carbon', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture']
        for param in soil_params:
            if param in row and param != parameter:
                popup_html += f"<b>{param}:</b> {row[param]:.2f}<br>"
        
        popup_html += "</div>"
        return popup_html
    
    def _add_soil_legend(self, m: folium.Map, parameter: str):
        """
        Add legend to soil quality map.
        
        Args:
            m: folium.Map object
            parameter: Soil parameter being visualized
        """
        if parameter == 'pH':
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 120px; 
                        border:2px solid grey; z-index:9999; font-size:12px;
                        background-color:white; padding:10px;">
                <b>pH Scale:</b><br>
                <i style="background:red; width:20px; height:20px; display:inline-block;"></i> Acidic (<5.5)<br>
                <i style="background:orange; width:20px; height:20px; display:inline-block;"></i> Slightly Acidic (5.5-6.5)<br>
                <i style="background:green; width:20px; height:20px; display:inline-block;"></i> Neutral (6.5-7.5)<br>
                <i style="background:blue; width:20px; height:20px; display:inline-block;"></i> Alkaline (>7.5)<br>
            </div>
            '''
        elif parameter == 'Organic_Carbon':
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 180px; height: 120px; 
                        border:2px solid grey; z-index:9999; font-size:12px;
                        background-color:white; padding:10px;">
                <b>Organic Carbon (%):</b><br>
                <i style="background:red; width:20px; height:20px; display:inline-block;"></i> Low (<0.8%)<br>
                <i style="background:orange; width:20px; height:20px; display:inline-block;"></i> Medium (0.8-1.5%)<br>
                <i style="background:yellow; width:20px; height:20px; display:inline-block;"></i> Good (1.5-2.5%)<br>
                <i style="background:green; width:20px; height:20px; display:inline-block;"></i> High (>2.5%)<br>
            </div>
            '''
        else:
            legend_html = f'''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 100px; 
                        border:2px solid grey; z-index:9999; font-size:12px;
                        background-color:white; padding:10px;">
                <b>{parameter}:</b><br>
                <i style="background:red; width:20px; height:20px; display:inline-block;"></i> Low<br>
                <i style="background:orange; width:20px; height:20px; display:inline-block;"></i> Medium<br>
                <i style="background:green; width:20px; height:20px; display:inline-block;"></i> High<br>
            </div>
            '''
        
        m.get_root().html.add_child(folium.Element(legend_html))

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    yield_data = pd.DataFrame({
        'Crop': np.random.choice(['Rice', 'Wheat', 'Maize', 'Sugarcane'], 100),
        'State': np.random.choice(['Punjab', 'Haryana', 'Uttar Pradesh'], 100),
        'District': np.random.choice(['Ludhiana', 'Hisar', 'Meerut'], 100),
        'Season': np.random.choice(['Kharif', 'Rabi'], 100),
        'Yield': np.random.uniform(1000, 5000, 100),
        'Area': np.random.uniform(1, 50, 100)
    })
    
    soil_data = pd.DataFrame({
        'Soil_Type': np.random.choice(['Clay', 'Sandy', 'Loamy'], 50),
        'pH': np.random.uniform(5.0, 8.5, 50),
        'Organic_Carbon': np.random.uniform(0.5, 3.0, 50),
        'Nitrogen': np.random.uniform(100, 500, 50)
    })
    
    # Initialize map generator
    map_gen = MapGenerator()
    
    # Create maps
    print("🗺️ Creating yield heatmap...")
    yield_map = map_gen.create_yield_heatmap(yield_data, crop_filter='Rice')
    
    print("🗺️ Creating soil quality map...")
    soil_map = map_gen.create_soil_quality_map(soil_data, soil_parameter='pH')
    
    print("📊 Creating yield comparison chart...")
    yield_chart = map_gen.create_yield_comparison_chart(yield_data, crops=['Rice', 'Wheat'])
    
    print("📈 Creating seasonal trend chart...")
    trend_chart = map_gen.create_seasonal_yield_trend(yield_data, crop='Rice')
    
    print("✅ Map generation complete!")