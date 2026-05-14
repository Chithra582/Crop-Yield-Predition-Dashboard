# chatbot.py - Advanced Agricultural AI Chatbot
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import time
from datetime import datetime
import re

class AgriculturalChatbot:
    """Advanced chatbot for agricultural Q&A with memory and context"""
    
    def __init__(self):
        self.initialize_chatbot()
        self.load_agriculture_knowledge()
        
    def initialize_chatbot(self):
        """Initialize chatbot session state"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'chat_context' not in st.session_state:
            st.session_state.chat_context = {
                'current_crop': None,
                'current_location': None,
                'user_expertise': 'beginner',
                'conversation_topic': 'general'
            }
        
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {
                'farm_size': None,
                'soil_type': None,
                'irrigation_type': None,
                'experience_years': None
            }
    
    def load_agriculture_knowledge(self):
        """Load agricultural knowledge base"""
        self.crop_knowledge = {
            'rice': {
                'description': 'Staple food crop requiring warm, wet conditions',
                'season': 'Kharif (June-September planting)',
                'temperature': '20-35°C optimal',
                'rainfall': '1000-2500 mm annually',
                'soil': 'Clayey or loamy with good water retention',
                'ph': '5.5-6.5',
                'duration': '3-6 months',
                'water': 'Requires flooded fields during growth',
                'fertilizer': 'N: 80-120 kg/ha, P: 40-60 kg/ha, K: 40-60 kg/ha',
                'yield': '4000-6000 kg/ha average',
                'diseases': ['Blast', 'Bacterial leaf blight', 'Sheath blight'],
                'pests': ['Brown plant hopper', 'Stem borer', 'Rice bug']
            },
            'wheat': {
                'description': 'Winter cereal crop, major staple food',
                'season': 'Rabi (October-November planting)',
                'temperature': '15-25°C optimal',
                'rainfall': '300-600 mm',
                'soil': 'Well-drained loamy soil',
                'ph': '6.0-7.5',
                'duration': '4-5 months',
                'water': 'Moderate, 4-5 irrigations',
                'fertilizer': 'N: 100-120 kg/ha, P: 60-80 kg/ha, K: 40-60 kg/ha',
                'yield': '3000-4500 kg/ha',
                'diseases': ['Rust', 'Smut', 'Powdery mildew'],
                'pests': ['Aphids', 'Army worm', 'Termites']
            },
            'maize': {
                'description': 'Versatile cereal for food and feed',
                'season': 'Kharif and Rabi',
                'temperature': '18-27°C',
                'rainfall': '500-800 mm',
                'soil': 'Deep, well-drained fertile soil',
                'ph': '5.8-7.0',
                'duration': '3-4 months',
                'water': 'Moderate, sensitive to waterlogging',
                'fertilizer': 'N: 120-150 kg/ha, P: 60-80 kg/ha, K: 60-80 kg/ha',
                'yield': '4000-6000 kg/ha',
                'diseases': ['Turcicum leaf blight', 'Maydis leaf blight'],
                'pests': ['Stem borer', 'Aphids', 'Earworm']
            },
            'cotton': {
                'description': 'Fiber crop for textile industry',
                'season': 'Kharif',
                'temperature': '25-35°C',
                'rainfall': '500-800 mm',
                'soil': 'Black cotton soil preferred',
                'ph': '6.0-8.0',
                'duration': '5-6 months',
                'water': 'Moderate, sensitive to waterlogging',
                'fertilizer': 'N: 100-150 kg/ha, P: 50-80 kg/ha, K: 50-80 kg/ha',
                'yield': '1500-2500 kg/ha (lint)',
                'diseases': ['Fusarium wilt', 'Verticillium wilt'],
                'pests': ['Bollworm', 'Whitefly', 'Aphids']
            },
            'sugarcane': {
                'description': 'Sugar-producing perennial grass',
                'season': 'Throughout year',
                'temperature': '20-30°C',
                'rainfall': '1500-2500 mm',
                'soil': 'Deep, well-drained loamy soil',
                'ph': '6.5-7.5',
                'duration': '10-12 months',
                'water': 'High water requirement',
                'fertilizer': 'N: 200-250 kg/ha, P: 80-120 kg/ha, K: 100-150 kg/ha',
                'yield': '70000-100000 kg/ha',
                'diseases': ['Red rot', 'Smut', 'Yellow leaf'],
                'pests': ['Early shoot borer', 'Top borer', 'Mealybug']
            }
        }
        
        self.general_knowledge = {
            'fertilizers': {
                'urea': 'Nitrogen source (46% N), apply before sowing or as top dressing',
                'dap': 'Diammonium phosphate (18% N, 46% P2O5), good for initial growth',
                'mop': 'Muriate of potash (60% K2O), for flowering and fruiting',
                'npk': 'Balanced fertilizer with N, P, K in various ratios',
                'organic': 'Compost, farmyard manure, vermicompost - improves soil health'
            },
            'pesticides': {
                'insecticides': 'Imidacloprid, Chlorpyrifos, Cypermethrin for insect control',
                'fungicides': 'Mancozeb, Carbendazim, Hexaconazole for fungal diseases',
                'herbicides': 'Glyphosate, 2,4-D for weed control',
                'bio_pesticides': 'Neem oil, Bacillus thuringiensis, Trichoderma'
            },
            'irrigation': {
                'flood': 'Traditional method, 50-60% efficiency',
                'drip': 'Water efficient (90%), suitable for fruits and vegetables',
                'sprinkler': '75-85% efficiency, good for field crops',
                'rainfed': 'Dependent on rainfall, requires drought-resistant varieties'
            },
            'soil_types': {
                'clay': 'Fine particles, high water retention, poor drainage',
                'sandy': 'Coarse particles, good drainage, low nutrients',
                'loam': 'Ideal mix of sand, silt, clay - best for most crops',
                'silt': 'Medium texture, fertile, holds moisture well'
            }
        }
        
        self.farming_practices = {
            'organic_farming': 'Avoids synthetic chemicals, uses compost, crop rotation, biological control',
            'precision_farming': 'Uses technology (GPS, sensors) for optimized input application',
            'conservation_agriculture': 'Minimum tillage, crop rotation, soil cover',
            'integrated_pest_management': 'Combines biological, cultural, and chemical methods',
            'crop_rotation': 'Growing different crops sequentially to improve soil health'
        }
    
    def process_user_message(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Update conversation context
        self.update_context(user_input)
        
        # Classify query type
        query_type = self.classify_query(user_input)
        
        # Generate response based on query type
        if query_type == 'crop_info':
            response = self.handle_crop_query(user_input)
        elif query_type == 'disease_pest':
            response = self.handle_disease_query(user_input)
        elif query_type == 'fertilizer':
            response = self.handle_fertilizer_query(user_input)
        elif query_type == 'yield_prediction':
            response = self.handle_yield_query(user_input)
        elif query_type == 'weather_climate':
            response = self.handle_weather_query(user_input)
        elif query_type == 'market_price':
            response = self.handle_market_query(user_input)
        elif query_type == 'farming_practice':
            response = self.handle_practice_query(user_input)
        elif query_type == 'greeting':
            response = self.handle_greeting()
        elif query_type == 'help':
            response = self.handle_help()
        else:
            response = self.handle_general_query(user_input)
        
        # Add to chat history
        self.add_to_history('user', user_input)
        self.add_to_history('assistant', response)
        
        return response
    
    def classify_query(self, text: str) -> str:
        """Classify user query type"""
        text_lower = text.lower()
        
        # Crop information queries
        crop_keywords = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'crop', 'plant', 'grow']
        if any(keyword in text_lower for keyword in crop_keywords):
            if any(word in text_lower for word in ['disease', 'pest', 'insect', 'fungus', 'sick']):
                return 'disease_pest'
            return 'crop_info'
        
        # Fertilizer and nutrient queries
        if any(word in text_lower for word in ['fertilizer', 'nutrient', 'npk', 'urea', 'manure']):
            return 'fertilizer'
        
        # Yield prediction queries
        if any(word in text_lower for word in ['yield', 'production', 'harvest', 'output', 'productivity']):
            return 'yield_prediction'
        
        # Weather and climate queries
        if any(word in text_lower for word in ['weather', 'rain', 'temperature', 'climate', 'humidity']):
            return 'weather_climate'
        
        # Market price queries
        if any(word in text_lower for word in ['price', 'market', 'sell', 'buy', 'rate', 'cost']):
            return 'market_price'
        
        # Farming practice queries
        if any(word in text_lower for word in ['organic', 'irrigation', 'rotation', 'practice', 'method']):
            return 'farming_practice'
        
        # Greetings
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return 'greeting'
        
        # Help requests
        if any(word in text_lower for word in ['help', 'what can you do', 'features', 'capabilities']):
            return 'help'
        
        return 'general'
    
    def handle_crop_query(self, query: str) -> str:
        """Handle crop-specific queries"""
        query_lower = query.lower()
        
        # Extract crop name from query
        crop_name = None
        for crop in self.crop_knowledge.keys():
            if crop in query_lower:
                crop_name = crop
                break
        
        if not crop_name:
            return "I can tell you about Rice, Wheat, Maize, Cotton, and Sugarcane. Which crop are you interested in?"
        
        # Get crop information
        crop_info = self.crop_knowledge[crop_name]
        
        # Check what specific information is being asked
        if 'season' in query_lower or 'when' in query_lower:
            response = f"**{crop_name.title()} Planting Season:**\n{crop_info['season']}"
        elif 'temperature' in query_lower or 'temp' in query_lower:
            response = f"**{crop_name.title()} Temperature Requirements:**\nOptimal: {crop_info['temperature']}"
        elif 'soil' in query_lower:
            response = f"**{crop_name.title()} Soil Requirements:**\n{crop_info['soil']}, pH: {crop_info['ph']}"
        elif 'water' in query_lower or 'irrigation' in query_lower:
            response = f"**{crop_name.title()} Water Requirements:**\n{crop_info['water']}"
        elif 'fertilizer' in query_lower or 'nutrient' in query_lower:
            response = f"**{crop_name.title()} Fertilizer Requirements:**\n{crop_info['fertilizer']}"
        elif 'yield' in query_lower:
            response = f"**{crop_name.title()} Yield:**\nAverage: {crop_info['yield']}"
        elif 'duration' in query_lower or 'how long' in query_lower:
            response = f"**{crop_name.title()} Growth Duration:**\n{crop_info['duration']}"
        else:
            # General crop information
            response = f"""**{crop_name.title()} - Complete Information:**

**Description:** {crop_info['description']}
**Season:** {crop_info['season']}
**Temperature:** {crop_info['temperature']}
**Rainfall:** {crop_info['rainfall']}
**Soil:** {crop_info['soil']} (pH: {crop_info['ph']})
**Duration:** {crop_info['duration']}
**Water:** {crop_info['water']}
**Fertilizer:** {crop_info['fertilizer']}
**Yield:** {crop_info['yield']}

**Common Diseases:** {', '.join(crop_info['diseases'])}
**Common Pests:** {', '.join(crop_info['pests'])}"""
        
        return response
    
    def handle_disease_query(self, query: str) -> str:
        """Handle disease and pest queries"""
        query_lower = query.lower()
        
        # Extract crop and disease info
        for crop, info in self.crop_knowledge.items():
            if crop in query_lower:
                # Check for specific diseases mentioned
                for disease in info['diseases']:
                    if disease.lower() in query_lower:
                        return self.get_disease_remedy(crop, disease)
                
                for pest in info['pests']:
                    if pest.lower() in query_lower:
                        return self.get_pest_remedy(crop, pest)
                
                # General disease/pest information for the crop
                response = f"""**{crop.title()} - Common Problems:**

**Diseases:**
{', '.join(info['diseases'])}

**Pests:**
{', '.join(info['pests'])}

You can ask about specific diseases or pests for more detailed information and remedies."""
                return response
        
        return "I can help with crop diseases and pests. Please mention which crop you're concerned about."
    
    def get_disease_remedy(self, crop: str, disease: str) -> str:
        """Get remedy for specific disease"""
        remedies = {
            'Blast': 'Use resistant varieties, apply Tricyclazole or Carbendazim fungicides',
            'Bacterial leaf blight': 'Use certified seeds, apply Streptomycin or Copper compounds',
            'Sheath blight': 'Apply Validamycin or Propiconazole, maintain proper spacing',
            'Rust': 'Use resistant varieties, apply Propiconazole or Tebuconazole',
            'Smut': 'Use treated seeds, apply Carboxin or Thiram',
            'Powdery mildew': 'Apply Sulfur dust or Dinocap, ensure good air circulation',
            'Turcicum leaf blight': 'Use resistant hybrids, apply Mancozeb or Chlorothalonil',
            'Maydis leaf blight': 'Crop rotation, apply Mancozeb or Propiconazole',
            'Fusarium wilt': 'Use resistant varieties, soil solarization, crop rotation',
            'Verticillium wilt': 'Crop rotation with non-host crops, soil fumigation',
            'Red rot': 'Use disease-free setts, hot water treatment, crop rotation',
            'Yellow leaf': 'Use resistant varieties, control aphid vectors'
        }
        
        remedy = remedies.get(disease, 'Consult local agricultural extension officer for specific recommendations')
        
        return f"""**{disease} in {crop.title()}**

**Symptoms:** Varies based on disease stage
**Prevention:** Use certified seeds, maintain field hygiene
**Treatment:** {remedy}
**Cultural Control:** Crop rotation, proper spacing, balanced fertilization"""
    
    def get_pest_remedy(self, crop: str, pest: str) -> str:
        """Get remedy for specific pest"""
        remedies = {
            'Brown plant hopper': 'Apply Imidacloprid or Buprofezin, release natural enemies',
            'Stem borer': 'Use pheromone traps, apply Chlorantraniliprole or Cartap',
            'Rice bug': 'Apply Cypermethrin or Lambda-cyhalothrin, hand picking',
            'Aphids': 'Apply Imidacloprid or Acetamiprid, release ladybugs',
            'Army worm': 'Apply Emamectin benzoate or Spinosad, manual removal',
            'Termites': 'Apply Chlorpyrifos or Fipronil soil treatment',
            'Earworm': 'Apply Emamectin benzoate or Indoxacarb, timely harvesting',
            'Bollworm': 'Use Bt cotton, apply Spinosad or Emamectin benzoate',
            'Whitefly': 'Apply Thiamethoxam or Pyriproxyfen, yellow sticky traps',
            'Early shoot borer': 'Apply Carbofuran or Chlorpyrifos granules',
            'Top borer': 'Apply Fipronil or Chlorantraniliprole, release Trichogramma',
            'Mealybug': 'Apply Buprofezin or Flonicamid, release Cryptolaemus beetles'
        }
        
        remedy = remedies.get(pest, 'Consult local agricultural extension officer for specific recommendations')
        
        return f"""**{pest} in {crop.title()}**

**Damage:** Varies based on pest lifecycle
**Prevention:** Regular monitoring, field sanitation
**Treatment:** {remedy}
**IPM:** Combine chemical, biological, and cultural methods"""
    
    def handle_fertilizer_query(self, query: str) -> str:
        """Handle fertilizer and nutrient queries"""
        query_lower = query.lower()
        
        # Check for specific fertilizer types
        for fert, info in self.general_knowledge['fertilizers'].items():
            if fert in query_lower:
                return f"""**{fert.upper()} Information:**

{info}

**Application Tips:**
- Test soil before application
- Apply at recommended rates
- Incorporate properly into soil
- Avoid contact with plant leaves"""
        
        # General fertilizer information
        return """**Fertilizer Guide:**

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
    
    def handle_yield_query(self, query: str) -> str:
        """Handle yield prediction queries"""
        query_lower = query.lower()
        
        # Check for specific crop
        for crop in self.crop_knowledge.keys():
            if crop in query_lower:
                crop_info = self.crop_knowledge[crop]
                
                return f"""**{crop.title()} Yield Improvement Tips:**

**Current Average:** {crop_info['yield']}

**To Increase Yield:**
1. **Soil Preparation:** Deep plowing, proper leveling
2. **Seed Quality:** Use certified, high-yielding varieties
3. **Planting Time:** Follow recommended sowing dates
4. **Spacing:** Maintain optimal plant population
5. **Nutrient Management:** Balanced fertilization based on soil test
6. **Water Management:** Timely irrigation, avoid water stress
7. **Pest Control:** Regular monitoring, integrated pest management
8. **Weed Control:** Timely weeding or herbicide application

**Expected Increase:** Following best practices can increase yield by 15-30%"""
        
        return """**General Yield Improvement Strategies:**

**Soil Health:**
- Regular soil testing
- Add organic matter
- Maintain proper pH

**Crop Management:**
- Select suitable varieties
- Optimize planting density
- Implement crop rotation

**Input Optimization:**
- Precision fertilizer application
- Efficient irrigation systems
- Timely pest control

**Technology Adoption:**
- Weather-based advisories
- Soil moisture sensors
- Drone-based monitoring"""
    
    def handle_weather_query(self, query: str) -> str:
        """Handle weather and climate queries"""
        query_lower = query.lower()
        
        # Simulated weather data
        weather_data = {
            'current': {
                'temperature': '25°C',
                'humidity': '65%',
                'rainfall': '0.0 mm',
                'wind': '12 km/h',
                'condition': 'Sunny'
            },
            'forecast': {
                'tomorrow': 'Partly cloudy, 24-28°C',
                'week': 'Mild with scattered showers',
                'month': 'Normal rainfall expected'
            }
        }
        
        if 'forecast' in query_lower or 'tomorrow' in query_lower or 'week' in query_lower:
            return f"""**Weather Forecast:**

**Tomorrow:** {weather_data['forecast']['tomorrow']}
**This Week:** {weather_data['forecast']['week']}
**This Month:** {weather_data['forecast']['month']}

**Farming Recommendations:**
- Good conditions for field preparation
- Monitor weather for irrigation scheduling
- Prepare for possible showers"""
        
        return f"""**Current Weather Conditions:**

**Temperature:** {weather_data['current']['temperature']}
**Humidity:** {weather_data['current']['humidity']}
**Rainfall:** {weather_data['current']['rainfall']}
**Wind Speed:** {weather_data['current']['wind']}
**Condition:** {weather_data['current']['condition']}

**Impact on Farming:**
- Good for most field operations
- Optimal for crop growth
- No irrigation needed today"""
    
    def handle_market_query(self, query: str) -> str:
        """Handle market price queries"""
        # Simulated market data
        market_prices = {
            'rice': '₹35-40/kg (Basmati), ₹25-30/kg (Normal)',
            'wheat': '₹22-25/kg',
            'maize': '₹18-20/kg',
            'cotton': '₹6500-7000/quintal',
            'sugarcane': '₹320-350/quintal',
            'potato': '₹15-20/kg',
            'tomato': '₹25-40/kg',
            'onion': '₹30-35/kg'
        }
        
        query_lower = query.lower()
        
        # Check for specific crop prices
        for crop, price in market_prices.items():
            if crop in query_lower:
                return f"""**{crop.title()} Market Prices:**

**Current Rate:** {price}

**Market Trends:**
- Steady demand in domestic market
- Export opportunities available
- Prices expected to remain stable

**Selling Tips:**
1. Monitor daily price fluctuations
2. Consider government procurement schemes
3. Explore direct market linkages
4. Store properly if expecting price rise"""
        
        # General market information
        response = "**Current Crop Market Prices:**\n\n"
        for crop, price in market_prices.items():
            response += f"**{crop.title()}:** {price}\n"
        
        response += "\n**Market Advice:**\n- Prices vary by region and quality\n- Check local mandi rates regularly\n- Consider contract farming for price stability"
        return response
    
    def handle_practice_query(self, query: str) -> str:
        """Handle farming practice queries"""
        query_lower = query.lower()
        
        # Check for specific practices
        for practice, info in self.farming_practices.items():
            if practice.replace('_', ' ') in query_lower:
                return f"""**{practice.replace('_', ' ').title()}:**

{info}

**Benefits:**
- Improved soil health
- Reduced input costs
- Sustainable production
- Better crop quality

**Implementation:**
- Start with small area
- Seek expert guidance
- Monitor results regularly"""
        
        if 'irrigation' in query_lower:
            return self.handle_irrigation_query()
        
        # General farming practices
        return """**Modern Farming Practices:**

**1. Organic Farming:**
   - No synthetic chemicals
   - Uses compost and green manure
   - Biological pest control

**2. Precision Agriculture:**
   - GPS-guided equipment
   - Soil sensor technology
   - Variable rate application

**3. Conservation Agriculture:**
   - Minimum tillage
   - Soil cover maintenance
   - Crop rotation

**4. Integrated Farming System:**
   - Combine crops, livestock, fisheries
   - Waste recycling
   - Multiple income sources"""
    
    def handle_irrigation_query(self) -> str:
        """Handle irrigation-specific queries"""
        return """**Irrigation Methods Comparison:**

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

**4. Rainfed Farming**
   - Efficiency: Depends on rainfall
   - Water Use: Natural
   - Cost: Very low
   - Best For: Drought-resistant crops

**Recommendation:** Choose based on crop, water availability, and budget."""
    
    def handle_greeting(self) -> str:
        """Handle greeting messages"""
        greetings = [
            "Hello! I'm your Agricultural AI Assistant. How can I help you with farming today? 🌾",
            "Hi there! Ready to discuss crops, soil, weather, or farming practices?",
            "Welcome! I'm here to provide expert advice on agriculture. What would you like to know?",
            "Greetings! I can help with crop information, pest control, fertilizer advice, and more!"
        ]
        
        # Add context if available
        greeting = np.random.choice(greetings)
        
        if st.session_state.chat_context['current_crop']:
            greeting += f"\n\nLast we discussed {st.session_state.chat_context['current_crop']}. Would you like to continue with that?"
        
        return greeting
    
    def handle_help(self) -> str:
        """Handle help requests"""
        return """**🤖 How I Can Help You:**

**🌾 Crop Information:**
- Growing requirements for specific crops
- Planting seasons and methods
- Yield expectations and improvement

**🐛 Pest & Disease Management:**
- Identify common problems
- Recommended treatments
- Preventive measures

**🧪 Fertilizer & Nutrition:**
- Nutrient requirements
- Fertilizer selection
- Application methods

**💧 Water Management:**
- Irrigation scheduling
- Water conservation
- System selection

**📈 Yield Optimization:**
- Best practices
- Technology adoption
- Market information

**🌤️ Weather & Climate:**
- Weather forecasts
- Climate impact
- Seasonal planning

**💬 Just ask me anything about agriculture!**"""
    
    def handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        # Try to extract intent
        if 'thank' in query.lower():
            return "You're welcome! Feel free to ask if you have more questions about agriculture. Happy farming! 🌱"
        
        if 'who are you' in query.lower() or 'what are you' in query.lower():
            return "I'm an Agricultural AI Assistant trained to help farmers with crop management, pest control, fertilizer advice, and farming best practices."
        
        if 'contact' in query.lower() or 'help' in query.lower():
            return """**For Further Assistance:**

**Government Resources:**
- Krishi Vigyan Kendra (KVK)
- Agricultural Extension Officers
- State Agricultural Departments

**Helplines:**
- Kisan Call Center: 1800-180-1551
- Crop Insurance: 1800-110-001

**Digital Platforms:**
- mKisan Portal
- Kisan Suvidha App
- eNAM for market prices"""
        
        # Default response
        return """I understand you're asking about agriculture. Could you be more specific about:

- Which crop you're interested in?
- Any particular problem you're facing?
- Information about fertilizers or irrigation?
- Market prices or weather forecasts?

I'm here to help with all aspects of farming! 🌾"""
    
    def update_context(self, user_input: str):
        """Update conversation context based on user input"""
        text_lower = user_input.lower()
        
        # Update current crop
        for crop in self.crop_knowledge.keys():
            if crop in text_lower:
                st.session_state.chat_context['current_crop'] = crop
                break
        
        # Update location if mentioned
        locations = ['punjab', 'maharashtra', 'karnataka', 'up', 'uttar pradesh', 'west bengal']
        for location in locations:
            if location in text_lower:
                st.session_state.chat_context['current_location'] = location
                break
    
    def add_to_history(self, role: str, message: str):
        """Add message to chat history"""
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({
            'role': role,
            'content': message,
            'time': timestamp
        })
        
        # Keep only last 50 messages
        if len(st.session_state.chat_history) > 50:
            st.session_state.chat_history = st.session_state.chat_history[-50:]
    
    def clear_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
        st.session_state.chat_context = {
            'current_crop': None,
            'current_location': None,
            'user_expertise': 'beginner',
            'conversation_topic': 'general'
        }
    
    def export_conversation(self):
        """Export conversation to JSON"""
        export_data = {
            'export_date': datetime.now().isoformat(),
            'user_profile': st.session_state.user_profile,
            'chat_context': st.session_state.chat_context,
            'conversation': st.session_state.chat_history
        }
        
        return json.dumps(export_data, indent=2)
    
    def get_conversation_summary(self):
        """Get summary of conversation"""
        if not st.session_state.chat_history:
            return "No conversation yet."
        
        crop_count = {}
        topics = set()
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                text = message['content'].lower()
                # Count crops mentioned
                for crop in self.crop_knowledge.keys():
                    if crop in text:
                        crop_count[crop] = crop_count.get(crop, 0) + 1
                
                # Identify topics
                if any(word in text for word in ['disease', 'pest']):
                    topics.add('Pest & Disease')
                elif any(word in text for word in ['fertilizer', 'nutrient']):
                    topics.add('Fertilizer')
                elif any(word in text for word in ['water', 'irrigation']):
                    topics.add('Irrigation')
                elif any(word in text for word in ['yield', 'production']):
                    topics.add('Yield')
                elif any(word in text for word in ['price', 'market']):
                    topics.add('Market')
        
        summary = f"**Conversation Summary**\n\n"
        summary += f"**Total Messages:** {len(st.session_state.chat_history)}\n"
        
        if crop_count:
            main_crops = sorted(crop_count.items(), key=lambda x: x[1], reverse=True)[:3]
            summary += f"**Main Crops Discussed:** {', '.join([c[0].title() for c in main_crops])}\n"
        
        if topics:
            summary += f"**Topics Covered:** {', '.join(topics)}\n"
        
        if st.session_state.chat_context['current_crop']:
            summary += f"**Current Focus:** {st.session_state.chat_context['current_crop'].title()}\n"
        
        return summary
    
    def get_quick_responses(self) -> List[Dict[str, str]]:
        """Get list of quick response buttons"""
        return [
            {"icon": "🌾", "text": "Best time to plant rice?", "query": "When is the best time to plant rice?"},
            {"icon": "🐛", "text": "Common wheat diseases?", "query": "What are common diseases in wheat?"},
            {"icon": "🧪", "text": "Fertilizer for maize?", "query": "What fertilizer should I use for maize?"},
            {"icon": "💧", "text": "Irrigation methods?", "query": "What are the best irrigation methods?"},
            {"icon": "📈", "text": "Increase crop yield?", "query": "How can I increase my crop yield?"},
            {"icon": "💰", "text": "Current market prices?", "query": "What are current crop market prices?"}
        ]

# Streamlit Chatbot Interface Component
def chatbot_interface():
    """Main chatbot interface for Streamlit"""
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AgriculturalChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Title
    st.markdown("## 🤖 Crop AI Chatbot")
    st.markdown("### Your 24/7 Agricultural Advisor")
    
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat container
        chat_container = st.container(height=500, border=True)
        
        with chat_container:
            # Display chat history
            for message in chatbot.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px;'>
                        <div style='font-weight: bold; color: #1565c0;'>👤 You ({message['time']}):</div>
                        <div>{message['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 5px;'>
                        <div style='font-weight: bold; color: #2e7d32;'>🤖 Assistant ({message['time']}):</div>
                        <div>{message['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        user_input = st.text_input("💬 Ask me anything about crops:", key="chat_input", placeholder="Type your question here...")
        
        # Send button
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            if st.button("📤 Send Message", use_container_width=True, type="primary") and user_input:
                with st.spinner("Thinking..."):
                    response = chatbot.process_user_message(user_input)
                    st.rerun()
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                chatbot.clear_history()
                st.rerun()
    
    with col2:
        # Quick responses
        st.markdown("#### 🎯 Quick Questions")
        
        quick_responses = chatbot.get_quick_responses()
        for resp in quick_responses:
            if st.button(f"{resp['icon']} {resp['text']}", use_container_width=True):
                chatbot.process_user_message(resp['query'])
                st.rerun()
        
        st.markdown("---")
        
        # Conversation summary
        st.markdown("#### 📊 Conversation Summary")
        st.markdown(chatbot.get_conversation_summary())
        
        st.markdown("---")
        
        # Export options
        st.markdown("#### 💾 Export")
        
        if st.button("📥 Export Conversation", use_container_width=True):
            export_data = chatbot.export_conversation()
            st.download_button(
                label="Download JSON",
                data=export_data,
                file_name=f"crop_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Settings
        with st.expander("⚙️ Chatbot Settings"):
            expertise = st.selectbox(
                "Your Farming Experience",
                ["Beginner", "Intermediate", "Expert"],
                index=0
            )
            st.session_state.chat_context['user_expertise'] = expertise.lower()
            
            response_detail = st.select_slider(
                "Response Detail",
                options=["Brief", "Normal", "Detailed"],
                value="Normal"
            )
            
            st.caption("Settings apply to future responses")
    
    # Features section
    st.markdown("---")
    st.markdown("#### 🌟 What I Can Help With:")
    
    features = [
        ("🌾", "Crop Selection", "Best crops for your region and soil"),
        ("📅", "Farming Calendar", "Planting and harvesting schedules"),
        ("🧪", "Nutrient Management", "Fertilizer recommendations"),
        ("🐛", "Pest Control", "Identification and treatment"),
        ("💧", "Irrigation", "Water management strategies"),
        ("📈", "Yield Optimization", "Increase your crop production"),
        ("🌤️", "Weather Advice", "Climate impact on farming"),
        ("💰", "Market Insights", "Prices and selling strategies"),
        ("🔬", "Disease Diagnosis", "Identify and treat crop diseases"),
        ("♻️", "Sustainable Farming", "Eco-friendly practices")
    ]
    
    cols = st.columns(5)
    for idx, (icon, title, desc) in enumerate(features):
        with cols[idx % 5]:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; border-radius: 8px; background: #f5f5f5;'>
                <div style='font-size: 24px;'>{icon}</div>
                <div style='font-weight: bold;'>{title}</div>
                <div style='font-size: 12px; color: #666;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# For direct execution
if __name__ == "__main__":
    chatbot_interface()