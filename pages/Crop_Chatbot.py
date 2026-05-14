# pages/5_🤖_Crop_Chatbot.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import random
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crop Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chatbot-header {
        background: linear-gradient(90deg, #9C27B0, #673AB7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .chat-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(156, 39, 176, 0.1);
        height: 500px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #E1BEE7, #CE93D8);
        padding: 12px 15px;
        border-radius: 15px 15px 3px 15px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid #9C27B0;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #F3E5F5, #E1BEE7);
        padding: 12px 15px;
        border-radius: 15px 15px 15px 3px;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #7B1FA2;
    }
    
    .quick-question {
        background: #F3E5F5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border: 1px solid #9C27B0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .quick-question:hover {
        background: #E1BEE7;
        transform: translateY(-2px);
    }
    
    .expert-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Agricultural knowledge base
agriculture_knowledge = {
    # Crop information
    "rice": {
        "description": "Staple food crop requiring warm, wet conditions",
        "season": "Kharif (June-September planting)",
        "temperature": "20-35°C optimal",
        "rainfall": "1000-2500 mm annually",
        "soil": "Clayey or loamy with good water retention",
        "ph": "5.5-6.5",
        "water": "Requires flooded fields during growth",
        "fertilizer": "N: 80-120 kg/ha, P: 40-60 kg/ha, K: 40-60 kg/ha",
        "yield": "4000-6000 kg/ha average",
        "diseases": ["Blast", "Bacterial leaf blight", "Sheath blight"],
        "pests": ["Brown plant hopper", "Stem borer", "Rice bug"]
    },
    "wheat": {
        "description": "Winter cereal crop, major staple food",
        "season": "Rabi (October-November planting)",
        "temperature": "15-25°C optimal",
        "rainfall": "300-600 mm",
        "soil": "Well-drained loamy soil",
        "ph": "6.0-7.5",
        "water": "Moderate, 4-5 irrigations",
        "fertilizer": "N: 100-120 kg/ha, P: 60-80 kg/ha, K: 40-60 kg/ha",
        "yield": "3000-4500 kg/ha",
        "diseases": ["Rust", "Smut", "Powdery mildew"],
        "pests": ["Aphids", "Army worm", "Termites"]
    },
    "maize": {
        "description": "Versatile cereal for food and feed",
        "season": "Kharif and Rabi",
        "temperature": "18-27°C",
        "rainfall": "500-800 mm",
        "soil": "Deep, well-drained fertile soil",
        "ph": "5.8-7.0",
        "water": "Moderate, sensitive to waterlogging",
        "fertilizer": "N: 120-150 kg/ha, P: 60-80 kg/ha, K: 60-80 kg/ha",
        "yield": "4000-6000 kg/ha",
        "diseases": ["Turcicum leaf blight", "Maydis leaf blight"],
        "pests": ["Stem borer", "Aphids", "Earworm"]
    },
    "cotton": {
        "description": "Fiber crop for textile industry",
        "season": "Kharif",
        "temperature": "25-35°C",
        "rainfall": "500-800 mm",
        "soil": "Black cotton soil preferred",
        "ph": "6.0-8.0",
        "water": "Moderate, sensitive to waterlogging",
        "fertilizer": "N: 100-150 kg/ha, P: 50-80 kg/ha, K: 50-80 kg/ha",
        "yield": "1500-2500 kg/ha (lint)",
        "diseases": ["Fusarium wilt", "Verticillium wilt"],
        "pests": ["Bollworm", "Whitefly", "Aphids"]
    },
    "sugarcane": {
        "description": "Sugar-producing perennial grass",
        "season": "Throughout year",
        "temperature": "20-30°C",
        "rainfall": "1500-2500 mm",
        "soil": "Deep, well-drained loamy soil",
        "ph": "6.5-7.5",
        "water": "High water requirement",
        "fertilizer": "N: 200-250 kg/ha, P: 80-120 kg/ha, K: 100-150 kg/ha",
        "yield": "70000-100000 kg/ha",
        "diseases": ["Red rot", "Smut", "Yellow leaf"],
        "pests": ["Early shoot borer", "Top borer", "Mealybug"]
    },
    
    # General knowledge
    "fertilizers": {
        "urea": "Nitrogen source (46% N), apply before sowing or as top dressing",
        "dap": "Diammonium phosphate (18% N, 46% P2O5), good for initial growth",
        "mop": "Muriate of potash (60% K2O), for flowering and fruiting",
        "organic": "Compost, farmyard manure, vermicompost - improves soil health"
    },
    
    "pesticides": {
        "insecticides": "Imidacloprid, Chlorpyrifos, Cypermethrin for insect control",
        "fungicides": "Mancozeb, Carbendazim, Hexaconazole for fungal diseases",
        "herbicides": "Glyphosate, 2,4-D for weed control",
        "bio_pesticides": "Neem oil, Bacillus thuringiensis, Trichoderma"
    },
    
    "irrigation": {
        "flood": "Traditional method, 50-60% efficiency",
        "drip": "Water efficient (90%), suitable for fruits and vegetables",
        "sprinkler": "75-85% efficiency, good for field crops",
        "rainfed": "Dependent on rainfall, requires drought-resistant varieties"
    },
    
    "soil_management": {
        "testing": "Test soil every 2-3 years for pH, N, P, K, organic matter",
        "improvement": "Add organic matter, maintain proper pH, ensure good drainage",
        "conservation": "Minimum tillage, crop rotation, cover cropping"
    }
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_context' not in st.session_state:
    st.session_state.user_context = {
        "current_crop": None,
        "user_expertise": "beginner",
        "location": None
    }

# Chatbot response generation
def generate_chatbot_response(user_input):
    """Generate response based on user input"""
    user_input_lower = user_input.lower()
    
    # Check for greetings
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
    if any(greeting in user_input_lower for greeting in greetings):
        return get_greeting_response()
    
    # Check for thanks
    if "thank" in user_input_lower:
        return "You're welcome! I'm here to help with all your agricultural questions. Feel free to ask more!"
    
    # Check for crop information
    for crop in ["rice", "wheat", "maize", "cotton", "sugarcane"]:
        if crop in user_input_lower:
            st.session_state.user_context["current_crop"] = crop
            return get_crop_information(crop, user_input_lower)
    
    # Check for specific topics
    if "fertilizer" in user_input_lower or "nutrient" in user_input_lower:
        return get_fertilizer_info(user_input_lower)
    
    if "pest" in user_input_lower or "disease" in user_input_lower:
        return get_pest_info(user_input_lower)
    
    if "water" in user_input_lower or "irrigation" in user_input_lower:
        return get_irrigation_info(user_input_lower)
    
    if "soil" in user_input_lower:
        return get_soil_info(user_input_lower)
    
    if "yield" in user_input_lower or "production" in user_input_lower:
        return get_yield_info(user_input_lower)
    
    if "price" in user_input_lower or "market" in user_input_lower:
        return get_market_info(user_input_lower)
    
    if "organic" in user_input_lower:
        return get_organic_farming_info()
    
    if "weather" in user_input_lower or "climate" in user_input_lower:
        return get_weather_info()
    
    # Default response
    return get_default_response(user_input_lower)

def get_greeting_response():
    """Get greeting response"""
    greetings = [
        "Hello! I'm your Agricultural AI Assistant. How can I help you with farming today? 🌾",
        "Hi there! Ready to discuss crops, soil, weather, or farming practices?",
        "Welcome! I'm here to provide expert advice on agriculture. What would you like to know?"
    ]
    
    if st.session_state.user_context["current_crop"]:
        return f"{random.choice(greetings)}\n\nLast we discussed {st.session_state.user_context['current_crop']}. Would you like to continue with that?"
    
    return random.choice(greetings)

def get_crop_information(crop, user_input):
    """Get crop-specific information"""
    crop_info = agriculture_knowledge.get(crop, {})
    
    if not crop_info:
        return f"I don't have detailed information about {crop}. I know about Rice, Wheat, Maize, Cotton, and Sugarcane."
    
    # Check what specific information is being asked
    if "season" in user_input or "when" in user_input:
        return f"**{crop.title()} Planting Season:**\n{crop_info.get('season', 'Not specified')}"
    
    elif "temperature" in user_input or "temp" in user_input:
        return f"**{crop.title()} Temperature Requirements:**\nOptimal: {crop_info.get('temperature', 'Not specified')}"
    
    elif "soil" in user_input:
        return f"**{crop.title()} Soil Requirements:**\n{crop_info.get('soil', 'Not specified')}, pH: {crop_info.get('ph', 'Not specified')}"
    
    elif "water" in user_input or "irrigation" in user_input:
        return f"**{crop.title()} Water Requirements:**\n{crop_info.get('water', 'Not specified')}"
    
    elif "fertilizer" in user_input or "nutrient" in user_input:
        return f"**{crop.title()} Fertilizer Requirements:**\n{crop_info.get('fertilizer', 'Not specified')}"
    
    elif "yield" in user_input:
        return f"**{crop.title()} Yield:**\nAverage: {crop_info.get('yield', 'Not specified')}"
    
    elif "disease" in user_input or "pest" in user_input:
        diseases = crop_info.get('diseases', [])
        pests = crop_info.get('pests', [])
        
        response = f"**{crop.title()} - Common Problems:**\n\n"
        response += f"**Diseases:** {', '.join(diseases) if diseases else 'Not specified'}\n"
        response += f"**Pests:** {', '.join(pests) if pests else 'Not specified'}\n\n"
        response += "You can ask about specific diseases or pests for more information."
        return response
    
    else:
        # General crop information
        response = f"""**{crop.title()} - Complete Information:**

**Description:** {crop_info.get('description', 'Not specified')}
**Season:** {crop_info.get('season', 'Not specified')}
**Temperature:** {crop_info.get('temperature', 'Not specified')}
**Rainfall:** {crop_info.get('rainfall', 'Not specified')}
**Soil:** {crop_info.get('soil', 'Not specified')} (pH: {crop_info.get('ph', 'Not specified')})
**Water:** {crop_info.get('water', 'Not specified')}
**Fertilizer:** {crop_info.get('fertilizer', 'Not specified')}
**Yield:** {crop_info.get('yield', 'Not specified')}

**Common Diseases:** {', '.join(crop_info.get('diseases', [])) if crop_info.get('diseases') else 'Not specified'}
**Common Pests:** {', '.join(crop_info.get('pests', [])) if crop_info.get('pests') else 'Not specified'}"""
        
        return response

def get_fertilizer_info(user_input):
    """Get fertilizer information"""
    response = """**Fertilizer Guide:**

**Nitrogen (N):** Promotes leaf growth, green color
- Sources: Urea, Ammonium sulfate, CAN
- Deficiency: Yellow leaves, stunted growth

**Phosphorus (P):** Supports root development, flowering
- Sources: DAP, SSP, Rock phosphate
- Deficiency: Purple leaves, poor flowering

**Potassium (K):** Improves disease resistance, fruit quality
- Sources: MOP, SOP, Potassium nitrate
- Deficiency: Brown leaf edges, weak stems

**Recommendations:**
1. Test soil every 2-3 years
2. Apply based on crop requirements
3. Use split applications for better efficiency
4. Consider organic alternatives for soil health"""
    
    return response

def get_pest_info(user_input):
    """Get pest and disease information"""
    response = """**Integrated Pest Management (IPM):**

**1. Prevention:**
- Use resistant varieties
- Maintain field hygiene
- Proper crop rotation

**2. Monitoring:**
- Regular field inspections
- Use pheromone traps
- Weather-based alerts

**3. Control Methods:**
- **Cultural:** Proper spacing, timely sowing
- **Biological:** Natural predators, biopesticides
- **Chemical:** Targeted pesticide application

**Common Solutions:**
- Neem oil for general pests
- Trichoderma for fungal diseases
- Bt (Bacillus thuringiensis) for caterpillars"""
    
    return response

def get_irrigation_info(user_input):
    """Get irrigation information"""
    response = """**Irrigation Methods:**

**1. Flood Irrigation (Traditional)**
- Efficiency: 50-60%
- Water Use: High
- Cost: Low
- Best For: Rice, some field crops

**2. Drip Irrigation (Modern)**
- Efficiency: 90-95%
- Water Use: Low
- Cost: High initial
- Best For: Fruits, vegetables, orchards

**3. Sprinkler Irrigation**
- Efficiency: 75-85%
- Water Use: Medium
- Cost: Medium
- Best For: Field crops, gardens

**Water Conservation Tips:**
- Schedule irrigation based on soil moisture
- Use mulching to reduce evaporation
- Repair leaks promptly
- Consider rainwater harvesting"""
    
    return response

def get_soil_info(user_input):
    """Get soil information"""
    response = """**Soil Management:**

**Soil Testing:**
- Test every 2-3 years
- Check pH, N, P, K, organic matter
- Sample from multiple locations

**Soil Improvement:**
- Add organic matter (compost, manure)
- Maintain pH 6.0-7.0 (add lime if acidic, sulfur if alkaline)
- Ensure good drainage
- Practice minimum tillage

**Soil Types:**
- **Clay:** High water retention, poor drainage
- **Sandy:** Good drainage, low nutrients
- **Loam:** Ideal mix - best for most crops
- **Silt:** Fertile, holds moisture well"""
    
    return response

def get_yield_info(user_input):
    """Get yield improvement information"""
    response = """**Yield Improvement Strategies:**

**1. Soil Health:**
- Regular soil testing
- Balanced fertilization
- Organic matter addition

**2. Crop Management:**
- Select suitable varieties
- Optimal planting density
- Timely operations

**3. Water Management:**
- Efficient irrigation
- Rainwater harvesting
- Mulching for moisture retention

**4. Pest Management:**
- Integrated pest management
- Regular monitoring
- Timely interventions

**Expected Improvements:**
- Following best practices: 15-30% increase
- Technology adoption: Additional 10-20%
- Precision farming: Up to 50% improvement"""
    
    return response

def get_market_info(user_input):
    """Get market information"""
    response = """**Market Information:**

**Current Prices (Approximate):**
- Rice: ₹35-40/kg (Basmati), ₹25-30/kg (Normal)
- Wheat: ₹22-25/kg
- Maize: ₹18-20/kg
- Cotton: ₹6500-7000/quintal
- Sugarcane: ₹320-350/quintal

**Market Trends:**
- Steady demand for staple crops
- Growing organic market
- Export opportunities increasing

**Selling Tips:**
1. Monitor daily price fluctuations
2. Consider government procurement schemes
3. Explore direct market linkages
4. Store properly if expecting price rise
5. Consider contract farming for stability"""
    
    return response

def get_organic_farming_info():
    """Get organic farming information"""
    response = """**Organic Farming:**

**Principles:**
1. No synthetic chemicals
2. Soil health focus
3. Biodiversity promotion
4. Ecological balance

**Practices:**
- Compost and green manure
- Crop rotation
- Biological pest control
- Natural nutrient sources

**Benefits:**
- Better soil health long-term
- Reduced input costs
- Premium market prices
- Environmental sustainability

**Certification:**
- Required for organic labeling
- 3-year conversion period
- Regular inspections
- Record keeping essential"""
    
    return response

def get_weather_info():
    """Get weather information"""
    response = """**Weather & Climate:**

**Current Conditions:**
- Temperature: 25°C
- Humidity: 65%
- Rainfall: 0.0 mm
- Wind: 12 km/h
- Condition: Sunny

**Forecast:**
- Tomorrow: Partly cloudy, 24-28°C
- This Week: Mild with scattered showers
- This Month: Normal rainfall expected

**Farming Impact:**
- Good conditions for field preparation
- Monitor weather for irrigation scheduling
- Prepare for possible showers
- Ideal for most crop operations"""
    
    return response

def get_default_response(user_input):
    """Get default response for unknown queries"""
    responses = [
        "I understand you're asking about agriculture. Could you be more specific about which crop or topic you're interested in?",
        "I can help with crop information, pest control, fertilizer advice, irrigation methods, and farming practices. What specifically would you like to know?",
        "I'm trained to answer questions about Rice, Wheat, Maize, Cotton, Sugarcane, and general farming practices. How can I assist you?",
        "For agricultural advice, please mention the crop name or specific topic like 'fertilizer', 'irrigation', 'pest control', or 'soil management'."
    ]
    
    return random.choice(responses)

# Quick questions
quick_questions = [
    {"icon": "🌾", "question": "Best time to plant rice?", "response_key": "rice_season"},
    {"icon": "🐛", "question": "Common wheat diseases?", "response_key": "wheat_diseases"},
    {"icon": "🧪", "question": "Fertilizer for maize?", "response_key": "maize_fertilizer"},
    {"icon": "💧", "question": "Irrigation methods?", "response_key": "irrigation_methods"},
    {"icon": "📈", "question": "Increase crop yield?", "response_key": "yield_improvement"},
    {"icon": "💰", "question": "Current market prices?", "response_key": "market_prices"},
    {"icon": "🌱", "question": "Organic farming?", "response_key": "organic_farming"},
    {"icon": "🌡️", "question": "Weather impact?", "response_key": "weather_impact"}
]

# Main Chatbot Content
st.markdown('<h1 class="chatbot-header">🤖 CROP AI CHATBOT</h1>', unsafe_allow_html=True)
st.markdown("### Your 24/7 Agricultural Advisory System")

# Create main layout
col_chat, col_sidebar = st.columns([2, 1])

with col_chat:
    # Chat container
    st.markdown("### 💬 Chat with AI Assistant")
    
    chat_container = st.container(height=400)
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history[-20:]:  # Show last 20 messages
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div style="font-weight: bold; color: #7B1FA2;">👤 You</div>
                    <div>{message["content"]}</div>
                    <div style="font-size: 10px; color: #666; text-align: right;">{message.get("time", "")}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <div style="font-weight: bold; color: #4A148C;">🤖 Assistant</div>
                    <div>{message["content"]}</div>
                    <div style="font-size: 10px; color: #666;">{message.get("time", "")}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    col_input, col_send = st.columns([4, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your question:",
            placeholder="Ask about crops, soil, weather, farming practices...",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col_send:
        if st.button("📤 Send", use_container_width=True, type="primary") and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "time": datetime.now().strftime("%H:%M")
            })
            
            # Generate and add bot response
            with st.spinner("🤖 Thinking..."):
                response = generate_chatbot_response(user_input)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "time": datetime.now().strftime("%H:%M")
                })
            
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with col_sidebar:
    # Quick questions
    st.markdown("### 🎯 Quick Questions")
    
    for q in quick_questions:
        if st.button(f"{q['icon']} {q['question']}", 
                    use_container_width=True,
                    key=f"quick_{q['question']}"):
            # Map quick questions to responses
            if q['response_key'] == 'rice_season':
                response = "Rice is planted during Kharif season (June-September). Best time varies by region but generally with monsoon onset."
            elif q['response_key'] == 'wheat_diseases':
                response = "Common wheat diseases: Rust, Smut, Powdery mildew. Use resistant varieties and proper fungicide application."
            elif q['response_key'] == 'maize_fertilizer':
                response = "Maize needs N: 120-150 kg/ha, P: 60-80 kg/ha, K: 60-80 kg/ha. Apply based on soil test results."
            elif q['response_key'] == 'irrigation_methods':
                response = get_irrigation_info("irrigation")
            elif q['response_key'] == 'yield_improvement':
                response = get_yield_info("yield")
            elif q['response_key'] == 'market_prices':
                response = get_market_info("price")
            elif q['response_key'] == 'organic_farming':
                response = get_organic_farming_info()
            elif q['response_key'] == 'weather_impact':
                response = get_weather_info()
            else:
                response = "I can help with that! Please ask specifically."
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": q['question'],
                "time": datetime.now().strftime("%H:%M")
            })
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "time": datetime.now().strftime("%H:%M")
            })
            
            st.rerun()
    
    st.markdown("---")
    
    # Chat statistics
    st.markdown("#### 📊 Chat Statistics")
    st.metric("Total Messages", len(st.session_state.chat_history))
    
    if st.session_state.chat_history:
        last_time = st.session_state.chat_history[-1].get("time", "N/A")
        st.caption(f"Last message: {last_time}")
    
    # Export chat
    if st.button("📥 Export Chat", use_container_width=True):
        export_data = {
            "export_date": datetime.now().isoformat(),
            "chat_history": st.session_state.chat_history,
            "user_context": st.session_state.user_context
        }
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"crop_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    st.markdown("---")


# Features section
st.markdown("---")
st.markdown("### 🌟 **What I Can Help With:**")

features = [
    ("🌾", "Crop Selection", "Best crops for your region and soil type"),
    ("📅", "Farming Calendar", "Planting and harvesting schedules"),
    ("🧪", "Nutrient Management", "Fertilizer recommendations based on soil test"),
    ("🐛", "Pest Control", "Identification and integrated pest management"),
    ("💧", "Irrigation", "Water management strategies and scheduling"),
    ("📈", "Yield Optimization", "Increase your crop production with best practices"),
    ("🌤️", "Weather Advice", "Climate impact on farming operations"),
    ("💰", "Market Insights", "Current prices and selling strategies"),
    ("🔬", "Disease Diagnosis", "Identify and treat crop diseases"),
    ("♻️", "Sustainable Farming", "Eco-friendly and organic practices")
]

cols_features = st.columns(5)
for idx, (icon, title, description) in enumerate(features):
    with cols_features[idx % 5]:
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; border-radius: 10px; background: #F3E5F5; margin: 10px 0;'>
            <div style='font-size: 28px;'>{icon}</div>
            <div style='font-weight: bold; color: #4A148C;'>{title}</div>
            <div style='font-size: 12px; color: #666; margin-top: 5px;'>{description}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("🤖 **AI Agricultural Chatbot** | 24/7 Advisory Support | 🌾 **Expert Farming Guidance**")