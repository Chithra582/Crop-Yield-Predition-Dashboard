# pages/4_🎤_Voice_Assistant.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot import AgriculturalChatbot
from deep_translator import GoogleTranslator

# Page configuration
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .voice-header {
        background: linear-gradient(90deg, #FF4081, #F50057);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .voice-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255, 64, 129, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #2196F3;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #F1F8E9, #DCEDC8);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    
    .system-message {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF9800;
    }
    
    .voice-button {
        background: linear-gradient(135deg, #FF4081, #F50057) !important;
        color: white !important;
        border: none !important;
    }
    
    .listening-indicator {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'voice_conversation' not in st.session_state:
    st.session_state.voice_conversation = []

if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False

if 'voice_commands' not in st.session_state:
    st.session_state.voice_commands = []

LANGUAGE_CODES = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", 
    "Marathi": "mr", "Kannada": "kn", "Malayalam": "ml", 
    "Punjabi": "pa", "Gujarati": "gu", "Bengali": "bn"
}

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

if 'chatbot_instance' not in st.session_state:
    st.session_state.chatbot_instance = AgriculturalChatbot()

# Voice command database
voice_command_db = {
    # Navigation commands
    "go to dashboard": {"response": "Navigating to dashboard. Please click the dashboard tab.", "action": "navigate_dashboard"},
    "open predictor": {"response": "Opening yield predictor. You can predict crop yields there.", "action": "navigate_predictor"},
    "show maps": {"response": "Opening maps. You can view crop suitability maps there.", "action": "navigate_maps"},
    "open chatbot": {"response": "Opening AI chatbot. You can ask agricultural questions there.", "action": "navigate_chatbot"},
    
    # Information commands
    "tell me about rice": {"response": "Rice needs warm temperature (20-35°C) and plenty of water. Plant in June-July, harvest in October-November.", "action": "info_crop"},
    "what is wheat": {"response": "Wheat is a winter cereal. Needs cool temperature (15-25°C), moderate water. Plant in October-November.", "action": "info_crop"},
    "how to grow maize": {"response": "Maize needs 18-27°C temperature. Can be grown in both seasons. Requires deep, fertile soil.", "action": "info_cultivation"},
    
    # Prediction commands
    "predict yield": {"response": "To predict yield, please go to the predictor page and enter your crop details.", "action": "predict_yield"},
    "recommend crop": {"response": "For crop recommendations, I need to know your soil type and climate conditions.", "action": "recommend_crop"},
    
    # Weather commands
    "what's the weather": {"response": "Current weather: 25°C, sunny, no rain expected. Good conditions for farming.", "action": "weather_info"},
    "check rainfall": {"response": "Average rainfall is 800 mm. No rain expected today. Good for field work.", "action": "weather_rain"},
    
    # Soil commands
    "analyze soil": {"response": "For soil analysis, I recommend testing pH and nutrient levels. Most crops prefer pH 6-7.", "action": "soil_analysis"},
    "soil nutrients": {"response": "Important soil nutrients are Nitrogen for growth, Phosphorus for roots, Potassium for health.", "action": "soil_nutrients"},
    
    # Help commands
    "what can you do": {"response": "I can help with crop information, yield prediction, weather updates, soil analysis, and navigation.", "action": "help"},
    "list commands": {"response": "You can say: 'predict yield', 'tell me about crops', 'check weather', 'analyze soil', or 'go to dashboard'.", "action": "help"},
    
    # Control commands
    "stop listening": {"response": "Stopping voice recognition. Click 'Start Listening' to resume.", "action": "stop_listening"},
    "clear conversation": {"response": "Clearing conversation history.", "action": "clear_conversation"},
    
    # Greetings
    "hello": {"response": "Hello! I'm your agricultural voice assistant. How can I help you today?", "action": "greeting"},
    "hi": {"response": "Hi there! Ready to help with your farming queries.", "action": "greeting"},
    "good morning": {"response": "Good morning! A great day for farming. How can I assist you?", "action": "greeting"},
    
    # Thanks
    "thank you": {"response": "You're welcome! Happy to help with your agricultural needs.", "action": "thanks"},
    "thanks": {"response": "You're welcome! Feel free to ask if you need more help.", "action": "thanks"},
}

# Simulated voice recognition (in production, use actual speech recognition)
def simulate_voice_recognition(command_text):
    """Simulate voice recognition - in production, replace with actual speech recognition"""
    time.sleep(1)  # Simulate processing time
    return command_text

def process_voice_command(command_text):
    """Process voice command and generate response"""
    # Translate input if needed
    lang_code = LANGUAGE_CODES.get(st.session_state.selected_language, 'en')
    
    try:
        if lang_code != 'en':
            translated_command = GoogleTranslator(source=lang_code, target='en').translate(command_text)
        else:
            translated_command = command_text
    except Exception as e:
        translated_command = command_text
        
    command_lower = translated_command.lower()
    
    # Add user message to conversation
    st.session_state.voice_conversation.append({
        "role": "user",
        "content": command_text,
        "time": datetime.now().strftime("%H:%M:%S")
    })
    
    # Store command
    st.session_state.voice_commands.append({
        "command": command_text,
        "time": datetime.now().isoformat()
    })
    
    # Find matching command
    response_en = None
    action = None
    
    for cmd_pattern, cmd_info in voice_command_db.items():
        if cmd_pattern in command_lower:
            response_en = cmd_info["response"]
            action = cmd_info["action"]
            break
    
    # If no specific command, use Agricultural Chatbot
    if not response_en:
        response_en = st.session_state.chatbot_instance.process_user_message(translated_command)
        action = "chatbot_query"
        
    # Translate response back
    try:
        if lang_code != 'en':
            response = GoogleTranslator(source='en', target=lang_code).translate(response_en)
        else:
            response = response_en
    except Exception as e:
        response = response_en
    
    # Add assistant response to conversation
    st.session_state.voice_conversation.append({
        "role": "assistant",
        "content": response,
        "time": datetime.now().strftime("%H:%M:%S")
    })
    
    # Execute action if needed
    if action == "stop_listening":
        st.session_state.is_listening = False
    elif action == "clear_conversation":
        st.session_state.voice_conversation = []
    
    return response, action

def start_listening_simulation():
    """Start simulated voice listening"""
    st.session_state.is_listening = True
    
    # Add system message
    st.session_state.voice_conversation.append({
        "role": "system",
        "content": "🎤 Started listening... Speak your command",
        "time": datetime.now().strftime("%H:%M:%S")
    })

def stop_listening():
    """Stop voice listening"""
    st.session_state.is_listening = False
    
    # Add system message
    st.session_state.voice_conversation.append({
        "role": "system",
        "content": "⏹️ Stopped listening",
        "time": datetime.now().strftime("%H:%M:%S")
    })

# Main Voice Assistant Content
st.markdown('<h1 class="voice-header">🎤 VOICE ASSISTANT</h1>', unsafe_allow_html=True)
st.markdown("### Voice-controlled Agricultural Assistant")

# Create main layout
col_main1, col_main2 = st.columns([1.5, 1])

with col_main1:
    # Voice controls
    st.markdown('<div class="voice-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Voice Controls")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("🎤 Start Listening", 
                    type="primary", 
                    use_container_width=True,
                    disabled=st.session_state.is_listening):
            start_listening_simulation()
            st.rerun()
    
    with control_col2:
        if st.button("⏹️ Stop Listening", 
                    use_container_width=True,
                    disabled=not st.session_state.is_listening):
            stop_listening()
            st.rerun()
    
    with control_col3:
        if st.button("🗑️ Clear Chat", 
                    use_container_width=True):
            st.session_state.voice_conversation = []
            st.rerun()
    
    # Listening indicator
    if st.session_state.is_listening:
        st.markdown("""
        <div class="listening-indicator" style="text-align: center; padding: 10px;">
            <div style="font-size: 24px;">🎤</div>
            <div style="color: #FF4081; font-weight: bold;">LISTENING...</div>
            <div style="font-size: 12px; color: #666;">Speak your command clearly</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Start Listening' to begin voice commands")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Conversation display
    st.markdown("### 💬 Voice Conversation")
    
    conversation_container = st.container(height=400, border=False)
    
    with conversation_container:
        for message in st.session_state.voice_conversation[-10:]:  # Show last 10 messages
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div style="font-weight: bold; color: #1565C0;">
                        👤 You <small>({message["time"]})</small>
                    </div>
                    <div style="margin-top: 5px;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            elif message["role"] == "assistant":
                st.markdown(f"""
                <div class="assistant-message">
                    <div style="font-weight: bold; color: #2E7D32;">
                        🤖 Assistant <small>({message["time"]})</small>
                    </div>
                    <div style="margin-top: 5px;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            else:  # system
                st.markdown(f"""
                <div class="system-message">
                    <div style="font-weight: bold; color: #EF6C00;">
                        ⚙️ System <small>({message["time"]})</small>
                    </div>
                    <div style="margin-top: 5px;">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Manual input (fallback)
    st.markdown("---")
    st.markdown("#### ⌨️ Type Command (Fallback)")
    
    manual_command = st.text_input(
        "Type your voice command:",
        placeholder="e.g., 'Tell me about rice cultivation'",
        key="manual_input"
    )
    
    if st.button("📤 Send Typed Command", use_container_width=True) and manual_command:
        process_voice_command(manual_command)
        st.rerun()

with col_main2:
    # Quick voice commands
    st.markdown("### 🎯 Quick Commands")
    
    quick_commands = [
        {"icon": "🌾", "text": "Crop Info", "command": "Tell me about rice"},
        {"icon": "📈", "text": "Predict Yield", "command": "Predict yield"},
        {"icon": "🗺️", "text": "Show Maps", "command": "Show maps"},
        {"icon": "🌡️", "text": "Check Weather", "command": "What's the weather"},
        {"icon": "🧪", "text": "Soil Analysis", "command": "Analyze soil"},
        {"icon": "🤖", "text": "Open Chatbot", "command": "Open chatbot"}
    ]
    
    for cmd in quick_commands:
        if st.button(f"{cmd['icon']} {cmd['text']}", 
                    use_container_width=True,
                    help=f"Say: '{cmd['command']}'"):
            process_voice_command(cmd['command'])
            st.rerun()
    
    st.markdown("---")
    
    # Voice statistics
    st.markdown("#### 📊 Voice Statistics")
    
    total_commands = len(st.session_state.voice_commands)
    st.metric("Total Commands", total_commands)
    
    if st.session_state.voice_commands:
        last_command = st.session_state.voice_commands[-1]['command']
        st.caption(f"**Last Command:** {last_command[:40]}...")
    
    # Command categories
    st.markdown("#### 🗂️ Command Categories")
    
    categories = {
        "🌾 Crop Info": sum(1 for cmd in st.session_state.voice_commands 
                          if any(word in cmd['command'].lower() 
                                for word in ['rice', 'wheat', 'maize', 'crop'])),
        "📈 Prediction": sum(1 for cmd in st.session_state.voice_commands 
                           if any(word in cmd['command'].lower() 
                                for word in ['predict', 'yield', 'recommend'])),
        "🌤️ Weather": sum(1 for cmd in st.session_state.voice_commands 
                         if any(word in cmd['command'].lower() 
                               for word in ['weather', 'rain', 'temperature'])),
        "🧪 Soil": sum(1 for cmd in st.session_state.voice_commands 
                      if any(word in cmd['command'].lower() 
                            for word in ['soil', 'nutrient', 'ph'])),
    }
    
    for category, count in categories.items():
        if count > 0:
            st.write(f"{category}: {count}")
    
    st.markdown("---")
    
    # Voice settings
    st.markdown("#### ⚙️ Voice Settings")
    
    with st.form("voice_settings"):
        language = st.selectbox(
            "Language",
            list(LANGUAGE_CODES.keys()),
            index=list(LANGUAGE_CODES.keys()).index(st.session_state.selected_language) if st.session_state.selected_language in LANGUAGE_CODES else 0
        )
        
        voice_speed = st.select_slider(
            "Voice Speed",
            options=["Slow", "Normal", "Fast"],
            value="Normal"
        )
        
        auto_start = st.checkbox("Auto-start on page load", value=False)
        
        voice_feedback = st.checkbox("Voice responses", value=True)
        
        if st.form_submit_button("💾 Save Settings", use_container_width=True):
            st.session_state.selected_language = language
            st.success(f"Settings saved successfully! Language set to {language}.")

# Features section
st.markdown("---")
st.markdown("### 🎤 **How to Use Voice Assistant:**")

col_feat1, col_feat2, col_feat3 = st.columns(3)

with col_feat1:
    st.markdown("""
    #### 🎙️ **Getting Started:**
    1. Click **Start Listening**
    2. Speak clearly into microphone
    3. Wait for confirmation
    4. Speak your command
    5. Click **Stop** when done
    """)

with col_feat2:
    st.markdown("""
    #### 🗣️ **Best Practices:**
    • Use quiet environment
    • Speak in complete sentences
    • Mention crop names clearly
    • Use natural language
    • Pause between commands
    """)

with col_feat3:
    st.markdown("""
    #### 🔧 **Troubleshooting:**
    • Check microphone permissions
    • Ensure internet connection
    • Adjust microphone volume
    • Use typed commands if needed
    • Refresh page if issues persist
    """)

# What you can ask
st.markdown("---")
st.markdown("### 🌟 **What You Can Ask:**")

what_can_ask = [
    ("🌾", "Crop Information", "'Tell me about wheat cultivation'"),
    ("📈", "Yield Prediction", "'Predict rice yield for my farm'"),
    ("🗺️", "Location Advice", "'Best crops for Punjab region'"),
    ("🌡️", "Weather Queries", "'What's the weather forecast?'"),
    ("🧪", "Soil Analysis", "'How to test my soil pH?'"),
    ("💡", "Recommendations", "'Suggest crops for clay soil'"),
    ("📊", "Data Analysis", "'Show crop yield statistics'"),
    ("🔄", "Navigation", "'Go to dashboard' or 'Open predictor'")
]

cols_ask = st.columns(4)
for idx, (icon, title, example) in enumerate(what_can_ask):
    with cols_ask[idx % 4]:
        st.markdown(f"""
        <div style='padding: 15px; border-radius: 10px; background: #f8f9fa; margin: 10px 0; text-align: center;'>
            <div style='font-size: 24px;'>{icon}</div>
            <div style='font-weight: bold;'>{title}</div>
            <div style='font-size: 12px; color: #666; font-style: italic;'>{example}</div>
        </div>
        """, unsafe_allow_html=True)

# Simulated voice input (for demo purposes)
if st.session_state.is_listening:
    # In production, this would be actual voice recognition
    # For demo, we'll use a simulated approach
    
    # Simulate listening for a few seconds
    if 'voice_demo_timer' not in st.session_state:
        st.session_state.voice_demo_timer = time.time()
    
    elapsed = time.time() - st.session_state.voice_demo_timer
    
    if elapsed > 5:  # Every 5 seconds, simulate a voice command
        # Demo commands for simulation
        demo_commands = [
            "What's the best crop for Maharashtra?",
            "Tell me about organic farming",
            "How much water does sugarcane need?",
            "Check current weather conditions",
            "Predict yield for wheat"
        ]
        
        # Simulate processing a random command
        import random
        demo_command = random.choice(demo_commands)
        
        process_voice_command(demo_command)
        st.session_state.voice_demo_timer = time.time()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("🎤 **Voice Assistant** | Hands-free Agricultural Control | 🗣️ **Natural Language Processing**")