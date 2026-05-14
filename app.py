import streamlit as st
import random
import time
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Crop AI - Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
    }
    
    .hero {
        background: white;
        padding: 60px 40px;
        border-radius: 25px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-top: 5px solid #4CAF50;
    }
    
    .hero h1 {
        font-size: 48px;
        background: linear-gradient(45deg, #2E7D32, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        font-weight: 800;
    }
    
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background: #1E1E1E;
        color: white;
        padding: 12px 0;
        border-radius: 10px;
        margin-bottom: 30px;
        position: relative;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .ticker {
        display: inline-block;
        white-space: nowrap;
        padding-left: 100%;
        animation: ticker 30s linear infinite;
        font-family: monospace;
        font-size: 16px;
        font-weight: bold;
    }
    .ticker-item {
        display: inline-block;
        padding: 0 2rem;
    }
    .ticker-up { color: #4CAF50; }
    .ticker-down { color: #F44336; }
    .ticker-neutral { color: #9E9E9E; }
    
    @keyframes ticker {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-100%, 0, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Define Navigation items for easy programmatic access
navigation_items = {
    "Dashboard": "pages/Dashboard.py",
    "Predictor": "pages/Predictor.py",
    "Live Maps": "pages/Live_Maps.py",
    "Voice Assistant": "pages/Voice_Assistant.py",
    "Crop Chatbot": "pages/Crop_Chatbot.py"
}

def render_ticker():
    # Modern CSS Animated Live Market Ticker
    st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker">
            <div class="ticker-item">🔴 <b>LIVE MARKET PRICES:</b></div>
            <div class="ticker-item">WHEAT: ₹2,250/q <span class="ticker-up">(▲ 1.2%)</span></div>
            <div class="ticker-item">RICE (Basmati): ₹3,800/q <span class="ticker-down">(▼ 0.5%)</span></div>
            <div class="ticker-item">MAIZE: ₹1,950/q <span class="ticker-up">(▲ 2.1%)</span></div>
            <div class="ticker-item">COTTON: ₹6,900/q <span class="ticker-neutral">(— 0.0%)</span></div>
            <div class="ticker-item">SUGARCANE: ₹350/q <span class="ticker-up">(▲ 0.8%)</span></div>
            <div class="ticker-item">SOYBEAN: ₹4,100/q <span class="ticker-down">(▼ 1.1%)</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Live Market Ticker
    render_ticker()

    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1>🌾 Crop AI Platform</h1>
        <p style="font-size: 22px; color: #555; margin-bottom: 10px;">
            Next-Generation Intelligent Agriculture Solutions
        </p>
        <p style="font-size: 16px; color: #777; margin-bottom: 30px;">
            Empowering farmers with AI, IoT, Voice Control, and Real-Time Data Analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action Buttons to replace the broken HTML buttons
    st.write("### 🚀 Get Started")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Open Dashboard", use_container_width=True, type="primary"):
            st.switch_page(navigation_items["Dashboard"])
    with col2:
        if st.button("📈 AI Predictor", use_container_width=True, type="secondary"):
            st.switch_page(navigation_items["Predictor"])
    with col3:
        if st.button("🎤 Voice Assistant", use_container_width=True, type="secondary"):
            st.switch_page(navigation_items["Voice Assistant"])
    with col4:
        if st.button("🤖 Crop Chatbot", use_container_width=True, type="secondary"):
            st.switch_page(navigation_items["Crop Chatbot"])

    st.markdown("---")

    # Innovation Section: Live IoT Simulation
    st.write("### 📡 Live Field Sensors (Simulation)")
    st.info("Simulating real-time IoT sensor data streaming from the field.")
    
    sensor_col1, sensor_col2, sensor_col3, sensor_col4 = st.columns(4)
    
    # Simulate some live data variations
    moisture = random.uniform(45.0, 55.0)
    temp = random.uniform(22.0, 26.0)
    nitrogen = random.uniform(80.0, 95.0)
    ph = random.uniform(6.0, 7.0)

    with sensor_col1:
        st.metric(label="💧 Soil Moisture", value=f"{moisture:.1f}%", delta=f"{random.uniform(-1, 1):.1f}%")
    with sensor_col2:
        st.metric(label="🌡️ Soil Temperature", value=f"{temp:.1f}°C", delta=f"{random.uniform(-0.5, 0.5):.1f}°C", delta_color="inverse")
    with sensor_col3:
        st.metric(label="🌱 Nitrogen (N) Level", value=f"{nitrogen:.0f} mg/kg", delta=f"{random.uniform(-2, 2):.0f} mg/kg")
    with sensor_col4:
        st.metric(label="🧪 Soil pH", value=f"{ph:.2f}", delta=f"{random.uniform(-0.1, 0.1):.2f}")

    st.markdown("---")

    # Feature Cards Section (Replacing empty rectangle boxes)
    st.write("### ✨ Explore Features")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        with st.container(border=False):
            st.subheader("🗺️ Live Maps")
            st.write("Interactive satellite and geographical maps to monitor farmland health and weather patterns.")
            if st.button("View Maps", key="btn_maps"):
                st.switch_page(navigation_items["Live Maps"])
                
    with feat_col2:
        with st.container(border=False):
            st.subheader("🗣️ Multi-lingual Voice")
            st.write("Talk to your AI assistant in English, Hindi, or Tamil to get hands-free agricultural advice.")
            if st.button("Try Voice Command", key="btn_voice"):
                st.switch_page(navigation_items["Voice Assistant"])
                
    with feat_col3:
        with st.container(border=False):
            st.subheader("🧠 Deep Context AI")
            st.write("A chatbot that remembers your farm's context to provide accurate pest and fertilizer remedies.")
            if st.button("Chat Now", key="btn_chat"):
                st.switch_page(navigation_items["Crop Chatbot"])

if __name__ == "__main__":
    main()