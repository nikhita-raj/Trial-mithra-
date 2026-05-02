import pandas as pd
import re
import os
from datetime import datetime
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
from groq import Groq

from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class TrialMitraChatbot:
    def __init__(self):
        from database import get_engine
        self.engine = get_engine()
        # Read from SQLite trials table
        try:
            self.trials_df = pd.read_sql_table('trials', self.engine)
        except Exception:
            # Fallback if db isn't initialized
            self.trials_df = pd.read_csv('data/clinical_trials.csv')
            
        self.interaction_log = []
        self.translator = GoogleTranslator(source='auto', target='en')
        
        # Initialize Groq Client
        if not GROQ_API_KEY:
            print("Warning: GROQ_API_KEY is not set in the environment variables.")
        self.client = Groq(api_key=GROQ_API_KEY)
        
        # Prepare System Context with Trial Data
        trials_context = self.trials_df.to_string(index=False)
        self.system_prompt = f"""
        You are TrialMitra, a highly professional and empathetic AI Clinical Trial Assistant.
        Your goal is to provide structured, clear, and "Pro-Expert" answers using the following trial database:
        {trials_context}
        
        RULES FOR EXPERT FORMATTING:
        1. Always use **Bold Title** - followed by a concise, professional description.
        2. Use relevant emojis liberally to make the response engaging (e.g., 🩺, 🧪, 📈, 🏥, ✅).
        3. For lists of trials or benefits, use clear bullet points with emojis.
        4. If the user hasn't provided age/condition, ask for them politely.
        
        TONE:
        Empathetic, expert-level, and highly structured.
        """
    
    def check_basic_eligibility(self, age, bp, disease):
        """Basic eligibility check based on trial criteria (Used by UI)"""
        eligible_trials = []
        
        for _, trial in self.trials_df.iterrows():
            if (trial['disease'] == disease and 
                trial['min_age'] <= age <= trial['max_age'] and
                trial['min_bp'] <= bp <= trial['max_bp'] and
                trial['status'] in ['active', 'recruiting']):
                eligible_trials.append(trial)
        
        return eligible_trials
    
    def generate_response(self, user_input, patient_id=None, patient_name=None, age=None, bp=None, disease=None, selected_language=None):
        """Generate chatbot response with multilingual support and dynamic actions"""
        
        SUPPORTED_LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada', 'te': 'Telugu'}
        detected_lang = selected_language if selected_language in SUPPORTED_LANGUAGES else 'en'

        processing_input = user_input
        
        try:
            user_message = processing_input
            context_parts = []
            if patient_name: context_parts.append(f"Patient Name: {patient_name}")
            if patient_id: context_parts.append(f"Patient ID: {patient_id}")
            if age: context_parts.append(f"Age: {age}")
            if bp: context_parts.append(f"BP: {bp}")
            if disease: context_parts.append(f"Condition: {disease}")
            
            if context_parts:
                user_message += f"\n[Context: {', '.join(context_parts)}]"
            
            lang_name = SUPPORTED_LANGUAGES.get(detected_lang, 'English')
            dynamic_system_prompt = self.system_prompt + f"\n\nCRITICAL INSTRUCTION: You MUST provide your entire response in {lang_name}. Do not reply in English unless the requested language is English."
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": dynamic_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_tokens=2500,
            )
            raw_response = chat_completion.choices[0].message.content
            
            # Parse actions: [Action: X]
            actions = re.findall(r"\[Action:\s*(.*?)\]", raw_response)
            clean_response = re.sub(r"\[Action:\s*.*?\]", "", raw_response).strip()
            # Remove trailing commas/punctuation from cleaning
            clean_response = re.sub(r"[,.;:\s]+$", ".", clean_response)
            
        except Exception as e:
            print(f"LLM Error: {e}")
            clean_response = "I apologize, but I'm having trouble connecting to my AI brain right now. Please try again later."
            actions = ["Try again", "Check trials"]
            
        # 1. Parse actions FIRST before any translation (if LLM followed English prefix rule)
        # Handle both [Action: ] and translated variations like (ಕ್ರಮ: ) or [Action: ]
        suggested_actions = re.findall(r"(?:\[|\()(?:Action|ಕ್ರಮ|कार्य|చర్య|ಕ್ರಿಯೆ)[:：]\s*(.*?)(?:\]|\))", raw_response, re.IGNORECASE)
        
        # If no actions found with prefixes, try generic bracket/paren matching at the end
        if not suggested_actions:
            suggested_actions = re.findall(r"(?:\[|\()([^\:\]\)]*?[:：][^\]\)]+?)(?:\]|\))", raw_response)
        
        # Strip prefixes from found actions if they were caught in the generic match
        suggested_actions = [re.sub(r"^.*[:：]\s*", "", a).strip() for a in suggested_actions]
        
        # Clean the response text by removing all bracketed/parenthesized action tags
        clean_response = re.sub(r"(?:\[|\()(?:Action|ಕ್ರಮ|कार्य|చర్య|ಕ್ರಿಯೆ)[:：].*?(?:\]|\))", "", raw_response, flags=re.IGNORECASE).strip()
        # Also remove generic ones that look like actions at the end
        clean_response = re.sub(r"(?:\[|\()[^\]\)]*?[:：][^\]\)]+?(?:\]|\))", "", clean_response).strip()
        
        # Clean trailing punctuation
        clean_response = re.sub(r"[,.;:\s]+$", ".", clean_response)
        
        # 2. Failsafe Suggestions by Language
        if not suggested_actions:
            DEFAULTS = {
                'en': ["🔍 Find Trials", "✅ Check Eligibility", "📚 About Trials", "🤝 Join a Study", "🛡️ Safety Info", "💡 Benefits", "🏥 Specific Disease", "📞 Contact Info"],
                'kn': ["🔍 ಪ್ರಯೋಗಗಳನ್ನು ಹುಡುಕಿ", "✅ ಅರ್ಹತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ", "📚 ಪ್ರಯೋಗಗಳ ಬಗ್ಗೆ ತಿಳಿ", "🤝 ಅಧ್ಯಯನಕ್ಕೆ ಸೇರಿ", "🛡️ ಸುರಕ್ಷತಾ ಮಾಹಿತಿ", "💡 ಪ್ರಯೋಜನಗಳು", "🏥 ನಿರ್ದಿಷ್ಟ ಕಾಯಿಲೆ", "📞 ಸಂಪರ್ಕ ಮಾಹಿತಿ"],
                'hi': ["🔍 परीक्षण खोजें", "✅ पात्रता जांचें", "📚 परीक्षणों के बारे में", "🤝 अध्ययन में शामिल हों", "🛡️ सुरक्षा जानकारी", "💡 लाभ", "🏥 विशिष्ट रोग", "📞 संपर्क जानकारी"],
                'te': ["🔍 ట్రయల్స్ కనుగొనండి", "✅ అర్హతను తనిఖీ చేయండి", "📚 ట్రయల్స్ గురించి", "🤝 అధ్యయనంలో చేరండి", "🛡️ భద్రతా సమాచారం", "💡 ప్రయోజనాలు", "🏥 నిర్దిష్ట వ్యాధి", "📞 సంప్రదింపు సమాచారం"]
            }
            suggested_actions = DEFAULTS.get(detected_lang, DEFAULTS['en'])
            
        self.log_interaction(patient_id, user_input, "llm_query", language=detected_lang)
        
        return {
            "answer": clean_response,
            "suggestions": suggested_actions[:8] # Allow up to 8
        }
    
    def log_interaction(self, patient_id, user_input, intent, language='en'):
        """Log chatbot interaction"""
        interaction = {
            'chat_id': f'C{len(self.interaction_log) + 1:03d}',
            'patient_id': patient_id or 'UNKNOWN',
            'language': language,
            'user_input': user_input,
            'intent': intent,
            'timestamp': datetime.now()
        }
        
        self.interaction_log.append(interaction)
    
    def save_interactions(self):
        """Save interactions to database"""
        if self.interaction_log:
            df = pd.DataFrame(self.interaction_log)
            # Append to existing table
            df.to_sql('interactions', self.engine, if_exists='append', index=False)
            
            # Clear interaction log
            self.interaction_log.clear()
    
    def get_interaction_features(self, patient_id):
        """Extract features from patient's chatbot interactions using database"""
        try:
            query = f"SELECT * FROM interactions WHERE patient_id = '{patient_id}'"
            patient_interactions = pd.read_sql_query(query, self.engine)
        except Exception:
            # Fallback
            patient_interactions = pd.DataFrame()
            
            if len(patient_interactions) > 0:
                # Calculate engagement metrics
                total_interactions = len(patient_interactions)
                trial_inquiries = len(patient_interactions[patient_interactions['intent'] == 'trial_inquiry'])
                eligibility_checks = len(patient_interactions[patient_interactions['intent'] == 'eligibility_check'])
                
                # Trial interest score (1-5)
                trial_interest = min(5, max(1, (trial_inquiries + eligibility_checks) * 2))
                
                # Engagement level (1-5)
                engagement_level = min(5, max(1, total_interactions))
                
                return trial_interest, engagement_level

        return 3, 2

# Test the chatbot
if __name__ == "__main__":
    chatbot = TrialMitraChatbot()
    
    # Test interactions
    test_queries = [
        "What trials are available for diabetes?",
        "Am I eligible for heart disease study?",
        "I want to participate in cancer research",
        "Tell me about hypertension trials"
    ]
    
    print("=== TrialMitra Chatbot Test ===\n")
    
    for query in test_queries:
        print(f"User: {query}")
        response = chatbot.generate_response(query, patient_id="TEST001", age=45, bp=120, disease="diabetes")
        print(f"Bot: {response}\n")
        print("-" * 50 + "\n")
    
    chatbot.save_interactions()
    print("Interactions saved to chatbot_interactions.csv")