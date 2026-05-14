# voice_handler.py - Advanced Voice Recognition for Agriculture
import streamlit as st
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
from datetime import datetime
import json
import numpy as np
from typing import Optional, Dict, Any, List
import wave
import pyaudio
import tempfile
import os

class VoiceAssistant:
    """Advanced voice assistant for agricultural queries"""
    
    def __init__(self):
        self.initialize_voice_assistant()
        self.setup_audio_devices()
        self.load_voice_commands()
        
    def initialize_voice_assistant(self):
        """Initialize voice assistant session state"""
        if 'voice_messages' not in st.session_state:
            st.session_state.voice_messages = []
        
        if 'is_listening' not in st.session_state:
            st.session_state.is_listening = False
        
        if 'voice_settings' not in st.session_state:
            st.session_state.voice_settings = {
                'language': 'en-IN',
                'voice_gender': 'female',
                'speech_rate': 150,
                'volume': 0.8,
                'auto_listen': False,
                'voice_feedback': True
            }
        
        if 'voice_command_history' not in st.session_state:
            st.session_state.voice_command_history = []
    
    def setup_audio_devices(self):
        """Setup audio input/output devices"""
        try:
            self.recognizer = sr.Recognizer()
            self.engine = pyttsx3.init()
            
            # Get available microphones
            self.microphones = sr.Microphone.list_microphone_names()
            
            # Configure text-to-speech engine
            voices = self.engine.getProperty('voices')
            if st.session_state.voice_settings['voice_gender'] == 'female':
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            else:
                for voice in voices:
                    if 'male' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Set speech properties
            self.engine.setProperty('rate', st.session_state.voice_settings['speech_rate'])
            self.engine.setProperty('volume', st.session_state.voice_settings['volume'])
            
        except Exception as e:
            st.error(f"Audio setup error: {str(e)}")
            self.recognizer = None
            self.engine = None
    
    def load_voice_commands(self):
        """Load predefined voice commands"""
        self.voice_commands = {
            # Navigation commands
            "go to dashboard": {"action": "navigate", "target": "dashboard"},
            "show predictor": {"action": "navigate", "target": "predictor"},
            "open maps": {"action": "navigate", "target": "maps"},
            "show chatbot": {"action": "navigate", "target": "chatbot"},
            "open settings": {"action": "navigate", "target": "settings"},
            
            # Prediction commands
            "predict yield": {"action": "predict", "type": "yield"},
            "recommend crop": {"action": "predict", "type": "recommendation"},
            "analyze soil": {"action": "analyze", "type": "soil"},
            "check weather": {"action": "analyze", "type": "weather"},
            
            # Information commands
            "tell me about": {"action": "info", "type": "crop_info"},
            "what is": {"action": "info", "type": "definition"},
            "how to grow": {"action": "info", "type": "cultivation"},
            "fertilizer for": {"action": "info", "type": "fertilizer"},
            
            # Control commands
            "start listening": {"action": "control", "command": "start_listen"},
            "stop listening": {"action": "control", "command": "stop_listen"},
            "clear conversation": {"action": "control", "command": "clear"},
            "help": {"action": "control", "command": "help"},
            
            # Data commands
            "show data": {"action": "data", "command": "show"},
            "export data": {"action": "data", "command": "export"},
            "refresh data": {"action": "data", "command": "refresh"},
        }
        
        self.crop_keywords = [
            'rice', 'wheat', 'maize', 'cotton', 'sugarcane',
            'potato', 'tomato', 'apple', 'banana', 'grapes',
            'soybean', 'groundnut', 'barley', 'sunflower'
        ]
        
        self.action_keywords = {
            'predict': ['predict', 'forecast', 'estimate', 'calculate'],
            'info': ['tell', 'explain', 'describe', 'what is', 'how to'],
            'analyze': ['analyze', 'check', 'examine', 'evaluate'],
            'control': ['start', 'stop', 'clear', 'help', 'show', 'export']
        }
    
    def start_listening(self):
        """Start voice recognition in a separate thread"""
        if self.recognizer is None:
            st.error("Speech recognition not initialized")
            return False
        
        st.session_state.is_listening = True
        self.listening_thread = threading.Thread(target=self._listen_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
        return True
    
    def stop_listening(self):
        """Stop voice recognition"""
        st.session_state.is_listening = False
        if hasattr(self, 'listening_thread'):
            self.listening_thread.join(timeout=1)
    
    def _listen_loop(self):
        """Main listening loop"""
        with sr.Microphone() as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            st.session_state.voice_messages.append({
                "role": "system",
                "content": "🎤 Listening started... Speak now",
                "time": datetime.now().strftime("%H:%M:%S")
            })
            
            while st.session_state.is_listening:
                try:
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Recognize speech
                    text = self.recognizer.recognize_google(audio, language=st.session_state.voice_settings['language'])
                    
                    if text:
                        # Process recognized text
                        self.process_voice_command(text)
                        
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    st.session_state.voice_messages.append({
                        "role": "system",
                        "content": "❌ Could not understand audio. Please try again.",
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
                except sr.RequestError as e:
                    st.session_state.voice_messages.append({
                        "role": "system",
                        "content": f"⚠️ Speech recognition error: {str(e)}",
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
                except Exception as e:
                    st.session_state.voice_messages.append({
                        "role": "system",
                        "content": f"⚠️ Error: {str(e)}",
                        "time": datetime.now().strftime("%H:%M:%S")
                    })
    
    def process_voice_command(self, command_text: str):
        """Process recognized voice command"""
        command_text_lower = command_text.lower()
        
        # Add to history
        st.session_state.voice_messages.append({
            "role": "user",
            "content": command_text,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        
        # Store in command history
        st.session_state.voice_command_history.append({
            "command": command_text,
            "time": datetime.now().isoformat()
        })
        
        # Process command
        response = self.analyze_command(command_text_lower)
        
        # Add response to messages
        st.session_state.voice_messages.append({
            "role": "assistant",
            "content": response,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        
        # Speak response if enabled
        if st.session_state.voice_settings['voice_feedback']:
            self.speak(response)
        
        # Trigger Streamlit rerun
        st.rerun()
    
    def analyze_command(self, command: str) -> str:
        """Analyze voice command and generate response"""
        command_lower = command.lower()
        
        # Check for predefined commands
        for cmd_pattern, cmd_info in self.voice_commands.items():
            if cmd_pattern in command_lower:
                return self.execute_predefined_command(cmd_info, command_lower)
        
        # Check for crop-specific queries
        for crop in self.crop_keywords:
            if crop in command_lower:
                return self.handle_crop_query(crop, command_lower)
        
        # Check for action keywords
        for action_type, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in command_lower:
                    return self.handle_general_query(action_type, command_lower)
        
        # Default response
        return self.handle_unknown_command(command_lower)
    
    def execute_predefined_command(self, cmd_info: Dict, command: str) -> str:
        """Execute predefined voice command"""
        action = cmd_info['action']
        
        if action == 'navigate':
            target = cmd_info['target']
            return f"Navigating to {target}. Please click on the {target} tab in the sidebar."
        
        elif action == 'predict':
            pred_type = cmd_info['type']
            if pred_type == 'yield':
                return "Opening yield predictor. Please select a crop and enter parameters in the form."
            elif pred_type == 'recommendation':
                return "Starting crop recommendation. I'll analyze soil and climate conditions."
            else:
                return "Starting analysis. Please provide more details about what you want to analyze."
        
        elif action == 'info':
            info_type = cmd_info['type']
            if info_type == 'crop_info':
                # Extract crop name from command
                for crop in self.crop_keywords:
                    if crop in command:
                        return self.get_crop_info(crop)
                return "Which crop would you like information about? Please specify the crop name."
            else:
                return "I can provide information about crops, soil, fertilizers, and farming practices. Please be specific."
        
        elif action == 'control':
            control_cmd = cmd_info['command']
            if control_cmd == 'start_listen':
                return "I'm already listening. You can speak your commands."
            elif control_cmd == 'stop_listen':
                self.stop_listening()
                return "Stopped listening. Click the 'Start Listening' button to resume."
            elif control_cmd == 'clear':
                st.session_state.voice_messages = []
                return "Conversation cleared. Ready for new commands."
            elif control_cmd == 'help':
                return self.get_voice_help()
        
        elif action == 'data':
            data_cmd = cmd_info['command']
            if data_cmd == 'show':
                return "Showing data dashboard. Please check the data visualization section."
            elif data_cmd == 'export':
                return "Preparing data export. Please use the export button in the dashboard."
            elif data_cmd == 'refresh':
                return "Refreshing data from sources. This may take a moment."
        
        return "Command executed. Check the dashboard for results."
    
    def handle_crop_query(self, crop: str, command: str) -> str:
        """Handle crop-specific voice queries"""
        crop_info = {
            'rice': {
                'short': "Rice needs warm temperature (20-35°C) and plenty of water. Plant in June-July, harvest in October-November.",
                'detailed': "Rice is a Kharif crop requiring flooded fields. It grows in 3-6 months with proper care."
            },
            'wheat': {
                'short': "Wheat grows in cool weather (15-25°C). Plant in October-November, harvest in March-April.",
                'detailed': "Wheat is a Rabi crop needing moderate water. It requires well-drained loamy soil."
            },
            'maize': {
                'short': "Maize needs 18-27°C temperature. Can be grown in both Kharif and Rabi seasons.",
                'detailed': "Maize is a versatile cereal used for food and feed. It requires deep, fertile soil."
            },
            'cotton': {
                'short': "Cotton needs hot climate (25-35°C). Plant in May-June, harvest from October.",
                'detailed': "Cotton is a fiber crop preferring black cotton soil. It requires moderate water."
            }
        }
        
        info = crop_info.get(crop, {
            'short': f"{crop.title()} is an important agricultural crop.",
            'detailed': f"{crop.title()} has specific growing requirements based on climate and soil."
        })
        
        # Check what information is requested
        if 'how to grow' in command or 'cultivation' in command:
            return f"To grow {crop}: 1. Prepare soil well, 2. Use quality seeds, 3. Plant at right time, 4. Provide proper nutrients, 5. Control pests regularly."
        
        elif 'fertilizer' in command:
            return f"For {crop}, use balanced N-P-K fertilizer. Rice needs 80-120 kg N/ha, Wheat needs 100-120 kg N/ha."
        
        elif 'yield' in command or 'production' in command:
            return f"Average {crop} yield is 4000-6000 kg/ha. With good practices, you can increase yield by 20-30%."
        
        elif 'price' in command or 'market' in command:
            return f"Current {crop} market price varies by quality and region. Check local mandi rates for accurate prices."
        
        else:
            return f"{crop.title()}: {info['short']} Would you like more specific information about cultivation, fertilizer, or yield?"
    
    def handle_general_query(self, action_type: str, command: str) -> str:
        """Handle general voice queries"""
        if action_type == 'predict':
            if 'weather' in command:
                return "Current weather: 25°C, sunny, no rain expected. Good conditions for farming activities."
            elif 'soil' in command:
                return "Soil analysis requires soil testing. I recommend testing pH, N-P-K levels for accurate recommendations."
            else:
                return "I can help predict crop yield, recommend crops, or analyze conditions. Please be more specific."
        
        elif action_type == 'info':
            if 'soil' in command:
                return "Good soil should have proper texture, pH between 6-7, adequate organic matter, and good drainage."
            elif 'fertilizer' in command:
                return "Fertilizers provide essential nutrients: Nitrogen for growth, Phosphorus for roots, Potassium for health."
            elif 'pest' in command or 'disease' in command:
                return "For pest control, use integrated methods: cultural, biological, and chemical control combined."
            else:
                return "I can provide information about crops, soil, fertilizers, irrigation, and pest management."
        
        elif action_type == 'analyze':
            return "Analysis requires specific parameters. Please use the predictor page for detailed analysis with your data."
        
        elif action_type == 'control':
            return "You can control the system with commands like 'start listening', 'stop listening', 'clear conversation', or 'help'."
        
        return "I understand you want information. Please specify if it's about crops, soil, weather, or farming practices."
    
    def handle_unknown_command(self, command: str) -> str:
        """Handle unknown voice commands"""
        # Try to extract intent
        words = command.split()
        
        if len(words) < 2:
            return "I didn't understand. Please try again with a complete sentence."
        
        # Check for greeting
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if any(greeting in command for greeting in greetings):
            return "Hello! I'm your agricultural voice assistant. How can I help you with farming today?"
        
        # Check for thanks
        if 'thank' in command:
            return "You're welcome! Feel free to ask if you need more help."
        
        # Check for weather query
        if 'weather' in command or 'rain' in command or 'temperature' in command:
            return "Current weather is good for farming: 25°C, 65% humidity, sunny. No rain expected today."
        
        # Default response
        return f"I heard: '{command}'. I'm trained to help with agricultural queries. Try asking about crops, soil, weather, or farming practices."
    
    def get_crop_info(self, crop: str) -> str:
        """Get detailed crop information"""
        crop_info_db = {
            'rice': "Rice: Staple food crop. Needs warm temperature (20-35°C), plenty of water (flooded fields), clayey soil. Plant in June-July, harvest in October-November. Average yield: 4000-6000 kg/ha.",
            'wheat': "Wheat: Winter cereal. Needs cool temperature (15-25°C), moderate water, well-drained loamy soil. Plant in October-November, harvest in March-April. Average yield: 3000-4500 kg/ha.",
            'maize': "Maize: Versatile cereal. Grows at 18-27°C, needs moderate water, deep fertile soil. Can be grown in both seasons. Average yield: 4000-6000 kg/ha.",
            'cotton': "Cotton: Fiber crop. Needs hot climate (25-35°C), moderate water, black cotton soil. Plant in May-June, harvest from October. Average yield: 1500-2500 kg/ha lint.",
            'sugarcane': "Sugarcane: Sugar crop. Needs 20-30°C, high water, deep loamy soil. Year-round crop, 10-12 month duration. Average yield: 70000-100000 kg/ha."
        }
        
        return crop_info_db.get(crop, f"{crop.title()} is an important crop. For specific information, please check the crop encyclopedia in the dashboard.")
    
    def get_voice_help(self) -> str:
        """Get voice command help"""
        return """**Voice Commands Guide:**

**Navigation:**
- "Go to dashboard" - Main dashboard
- "Show predictor" - Yield prediction
- "Open maps" - Interactive maps
- "Show chatbot" - AI chat assistant

**Predictions:**
- "Predict yield" - Crop yield prediction
- "Recommend crop" - Crop recommendation
- "Analyze soil" - Soil analysis
- "Check weather" - Weather forecast

**Information:**
- "Tell me about [crop]" - Crop information
- "How to grow [crop]" - Cultivation guide
- "Fertilizer for [crop]" - Nutrient advice

**Control:**
- "Start listening" / "Stop listening"
- "Clear conversation"
- "Help" - Show this guide

Try saying: "Tell me about rice" or "Predict wheat yield" """
    
    def speak(self, text: str):
        """Convert text to speech"""
        if self.engine is None:
            return
        
        try:
            # Clean text for speech
            speech_text = text.replace('**', '').replace('\n', '. ')
            
            # Speak in a separate thread to avoid blocking
            def speak_thread():
                self.engine.say(speech_text)
                self.engine.runAndWait()
            
            threading.Thread(target=speak_thread, daemon=True).start()
            
        except Exception as e:
            st.error(f"Speech error: {str(e)}")
    
    def record_audio_sample(self, duration: int = 5):
        """Record audio sample for testing"""
        try:
            p = pyaudio.PyAudio()
            
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=44100,
                          input=True,
                          frames_per_buffer=1024)
            
            frames = []
            
            for _ in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wf = wave.open(f.name, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))
                wf.close()
                return f.name
        
        except Exception as e:
            st.error(f"Recording error: {str(e)}")
            return None
    
    def update_settings(self, settings: Dict):
        """Update voice settings"""
        st.session_state.voice_settings.update(settings)
        
        # Apply to TTS engine
        if self.engine:
            if 'speech_rate' in settings:
                self.engine.setProperty('rate', settings['speech_rate'])
            if 'volume' in settings:
                self.engine.setProperty('volume', settings['volume'])
            if 'voice_gender' in settings:
                voices = self.engine.getProperty('voices')
                if settings['voice_gender'] == 'female':
                    for voice in voices:
                        if 'female' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
                else:
                    for voice in voices:
                        if 'male' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
    
    def get_voice_statistics(self) -> Dict:
        """Get voice usage statistics"""
        total_commands = len(st.session_state.voice_command_history)
        
        # Analyze command types
        command_types = {'info': 0, 'predict': 0, 'control': 0, 'navigate': 0, 'data': 0}
        
        for cmd_entry in st.session_state.voice_command_history:
            cmd = cmd_entry['command'].lower()
            for cmd_type in command_types.keys():
                for pattern in self.voice_commands.keys():
                    if cmd_type in pattern and pattern in cmd:
                        command_types[cmd_type] += 1
                        break
        
        # Most common crops mentioned
        crop_mentions = {}
        for crop in self.crop_keywords:
            count = sum(1 for cmd_entry in st.session_state.voice_command_history 
                       if crop in cmd_entry['command'].lower())
            if count > 0:
                crop_mentions[crop] = count
        
        return {
            'total_commands': total_commands,
            'command_types': command_types,
            'crop_mentions': crop_mentions,
            'last_command': st.session_state.voice_command_history[-1]['command'] if st.session_state.voice_command_history else None,
            'first_command_date': st.session_state.voice_command_history[0]['time'] if st.session_state.voice_command_history else None
        }

# Streamlit Voice Interface Component
def voice_assistant_interface():
    """Main voice assistant interface for Streamlit"""
    
    # Initialize voice assistant
    if 'voice_assistant' not in st.session_state:
        st.session_state.voice_assistant = VoiceAssistant()
    
    assistant = st.session_state.voice_assistant
    
    # Title
    st.markdown("## 🎤 Voice Assistant")
    st.markdown("### Voice-controlled Agricultural Advisory System")
    
    # Create columns
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Status and controls
        st.markdown("#### 🎛️ Voice Controls")
        
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if st.button("🎤 Start Listening", 
                        type="primary", 
                        use_container_width=True,
                        disabled=st.session_state.is_listening):
                if assistant.start_listening():
                    st.success("Listening started!")
                    st.rerun()
        
        with control_col2:
            if st.button("⏹️ Stop Listening", 
                        use_container_width=True,
                        disabled=not st.session_state.is_listening):
                assistant.stop_listening()
                st.info("Listening stopped")
                st.rerun()
        
        with control_col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.voice_messages = []
                st.session_state.voice_command_history = []
                st.rerun()
        
        # Status indicator
        status_color = "🟢" if st.session_state.is_listening else "🔴"
        st.markdown(f"**Status:** {status_color} {'Listening...' if st.session_state.is_listening else 'Not listening'}")
        
        # Conversation display
        st.markdown("#### 💬 Voice Conversation")
        
        conversation_container = st.container(height=400, border=True)
        
        with conversation_container:
            for msg in st.session_state.voice_messages[-20:]:  # Show last 20 messages
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px; border-left: 4px solid #2196F3;'>
                        <div style='font-weight: bold; color: #1565c0;'>
                            👤 You <small>({msg["time"]})</small>
                        </div>
                        <div style='margin-top: 5px;'>{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif msg["role"] == "assistant":
                    st.markdown(f"""
                    <div style='background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px; border-left: 4px solid #4CAF50;'>
                        <div style='font-weight: bold; color: #2e7d32;'>
                            🤖 Assistant <small>({msg["time"]})</small>
                        </div>
                        <div style='margin-top: 5px;'>{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:  # system messages
                    st.markdown(f"""
                    <div style='background-color: #fff3e0; padding: 8px; border-radius: 8px; margin: 5px; border-left: 4px solid #FF9800;'>
                        <div style='font-weight: bold; color: #EF6C00;'>
                            ⚙️ System <small>({msg["time"]})</small>
                        </div>
                        <div style='margin-top: 5px; font-size: 0.9em;'>{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Manual input (fallback)
        st.markdown("---")
        st.markdown("#### ⌨️ Type Command (Fallback)")
        
        manual_input = st.text_input("Type command if voice isn't working:", 
                                   placeholder="e.g., 'Tell me about rice cultivation'")
        
        if st.button("📤 Send Typed Command", use_container_width=True) and manual_input:
            assistant.process_voice_command(manual_input)
            st.rerun()
    
    with col2:
        # Quick voice commands
        st.markdown("#### 🎯 Quick Commands")
        
        quick_commands = [
            {"icon": "🌾", "text": "Crop Info", "command": "Tell me about rice"},
            {"icon": "📈", "text": "Predict Yield", "command": "Predict wheat yield"},
            {"icon": "🗺️", "text": "Show Maps", "command": "Open maps"},
            {"icon": "🌡️", "text": "Check Weather", "command": "Check weather"},
            {"icon": "🧪", "text": "Soil Analysis", "command": "Analyze soil"},
            {"icon": "🤖", "text": "Open Chatbot", "command": "Show chatbot"}
        ]
        
        for cmd in quick_commands:
            if st.button(f"{cmd['icon']} {cmd['text']}", 
                        use_container_width=True,
                        help=f"Say: '{cmd['command']}'"):
                assistant.process_voice_command(cmd['command'])
                st.rerun()
        
        st.markdown("---")
        
        # Voice statistics
        st.markdown("#### 📊 Voice Statistics")
        
        stats = assistant.get_voice_statistics()
        st.metric("Total Commands", stats['total_commands'])
        
        if stats['last_command']:
            st.caption(f"**Last Command:** {stats['last_command'][:50]}...")
        
        with st.expander("View Detailed Stats"):
            st.write("**Command Types:**")
            for cmd_type, count in stats['command_types'].items():
                if count > 0:
                    st.write(f"- {cmd_type.title()}: {count}")
            
            if stats['crop_mentions']:
                st.write("**Most Mentioned Crops:**")
                for crop, count in list(stats['crop_mentions'].items())[:3]:
                    st.write(f"- {crop.title()}: {count}")
        
        st.markdown("---")
        
        # Voice settings
        st.markdown("#### ⚙️ Voice Settings")
        
        with st.form("voice_settings_form"):
            language = st.selectbox(
                "Language",
                ["en-IN (Indian English)", "en-US (US English)", "hi-IN (Hindi)", "ta-IN (Tamil)"],
                index=0
            )
            
            voice_gender = st.selectbox(
                "Voice Gender",
                ["Female", "Male"],
                index=0
            )
            
            speech_rate = st.slider(
                "Speech Rate",
                100, 300, 150,
                help="Words per minute"
            )
            
            volume = st.slider(
                "Volume",
                0.0, 1.0, 0.8, 0.1
            )
            
            auto_listen = st.checkbox(
                "Auto-start listening",
                value=False,
                help="Start listening when page loads"
            )
            
            voice_feedback = st.checkbox(
                "Voice responses",
                value=True,
                help="Assistant speaks responses"
            )
            
            if st.form_submit_button("💾 Save Settings", use_container_width=True):
                assistant.update_settings({
                    'language': language.split(' ')[0],
                    'voice_gender': voice_gender.lower(),
                    'speech_rate': speech_rate,
                    'volume': volume,
                    'auto_listen': auto_listen,
                    'voice_feedback': voice_feedback
                })
                st.success("Settings saved!")
    
    # Features and help section
    st.markdown("---")
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("#### 🎤 **How to Use:**")
        st.write("1. Click 'Start Listening'")
        st.write("2. Speak clearly into microphone")
        st.write("3. Use natural language commands")
        st.write("4. Click 'Stop' when done")
    
    with col_feat2:
        st.markdown("#### 🗣️ **Best Practices:**")
        st.write("• Speak in a quiet environment")
        st.write("• Use complete sentences")
        st.write("• Mention crop names clearly")
        st.write("• Wait for confirmation beep")
    
    with col_feat3:
        st.markdown("#### 🔧 **Troubleshooting:**")
        st.write("• Check microphone permissions")
        st.write("• Ensure internet connection")
        st.write("• Adjust microphone volume")
        st.write("• Use typed commands if needed")
    
    # Test microphone section
    st.markdown("---")
    st.markdown("#### 🎚️ Microphone Test")
    
    test_col1, test_col2 = st.columns([3, 1])
    
    with test_col1:
        test_duration = st.slider("Test recording duration (seconds)", 1, 10, 3)
    
    with test_col2:
        if st.button("🎙️ Test Mic", use_container_width=True):
            with st.spinner(f"Recording for {test_duration} seconds..."):
                audio_file = assistant.record_audio_sample(test_duration)
                if audio_file:
                    st.audio(audio_file, format='audio/wav')
                    os.unlink(audio_file)  # Clean up temp file
    
    # Voice capabilities
    st.markdown("---")
    st.markdown("#### 🌟 **What You Can Ask:**")
    
    capabilities = [
        ("🌾", "Crop Information", "'Tell me about wheat cultivation'"),
        ("📈", "Yield Prediction", "'Predict rice yield for my farm'"),
        ("🗺️", "Location Advice", "'Best crops for Punjab region'"),
        ("🌡️", "Weather Queries", "'What's the weather forecast?'"),
        ("🧪", "Soil Analysis", "'How to test my soil pH?'"),
        ("💡", "Recommendations", "'Suggest crops for clay soil'")
    ]
    
    cols = st.columns(3)
    for idx, (icon, title, example) in enumerate(capabilities):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style='padding: 15px; border-radius: 10px; background: #f8f9fa; margin: 10px 0;'>
                <div style='font-size: 24px; text-align: center;'>{icon}</div>
                <div style='font-weight: bold; text-align: center;'>{title}</div>
                <div style='font-size: 12px; color: #666; text-align: center; font-style: italic;'>{example}</div>
            </div>
            """, unsafe_allow_html=True)

# For direct execution
if __name__ == "__main__":
    voice_assistant_interface()