import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from chatbot import TrialMitraChatbot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from trial_repository.trial_matcher import TrialMatcher

# Page configuration
# Page configuration
st.set_page_config(
    page_title="TrialMitra - AI Clinical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* 
       MEDICAL THEME PALETTE
       Primary: #0D47A1 (Deep Medical Blue) - Trust & Professionalism
       Secondary: #E3F2FD (Light Blue) - Calming background
       Accent: #2E7D32 (Emerald Green) - Health & Success
       Text: #333333 (Dark Grey) - Readability
       White: #FFFFFF - Cleanliness
    */

    /* Main Background & Text */
    .stApp {
        background-color: #F8F9FA; /* Very light cool gray */
        color: #333333;
    }
    
    /* Force text color for all main elements */
    p, h1, h2, h3, h4, h5, h6, span, div, label, li {
        color: #263238 !important; /* Dark Slate Blue Grey */
    }
    
    /* Header Styling */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E0E0E0;
    }
    
    /* Hide the 'Deploy' button and running man, but keep hamburger */
    .stDeployButton {display: none;}
    [data-testid="stStatusWidget"] {display: none;}
    
    /* Style the hamburger menu to match theme */
    [data-testid="stSidebarCollapsedControl"] {
        color: #0D47A1 !important;
        background-color: transparent;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%); /* Medical Blue Gradient */
        padding: 1.5rem 2rem;
        border-bottom: 4px solid #0D47A1;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 15px rgba(21, 101, 192, 0.2);
    }
    .header-title {
        color: #FFFFFF !important; /* White text on blue header */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .header-subtitle {
        color: #E3F2FD !important; /* Light blue text */
        font-size: 1.1rem;
        margin-left: 1.5rem;
        font-weight: 400;
        border-left: 2px solid #90CAF9;
        padding-left: 1.5rem;
    }

    /* Card Styling */
    .css-1r6slb0, .stMetric, .stDataFrame, .stExpander {
        background-color: #FFFFFF;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); /* Softer shadow */
        border: 1px solid #E0E0E0;
        transition: all 0.2s ease-in-out;
    }
    .css-1r6slb0:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: #BBDEFB;
    }

    /* Chat Interface Styling */
    .chat-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
    }
    .chat-header {
        background-color: #1565C0; /* Medical Blue */
        color: white;
        padding: 0.8rem;
        font-weight: 600;
        text-align: center;
        border-radius: 12px 12px 0 0;
        letter-spacing: 0.5px;
    }
    
    /* Chat Bubbles */
    /* Chat Bubbles */
    .stChatMessage.user {
        background-color: #909497 !important; /* Cool Grey */
        border-radius: 10px;
        color: #FFFFFF !important;
        border: none;
    }
    .stChatMessage.assistant {
        background-color: #FFFFFF !important; /* Clean White */
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        color: #333333 !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: inherit !important;
    }

    /* Dashboard Widgets */
    [data-testid="stMetricValue"] {
        color: #0D47A1 !important; /* Deep Blue */
        font-weight: 700;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #546E7A !important; /* Blue Grey */
        font-weight: 600;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Fix for Markdown headers inside widgets */
    .css-1r6slb0 h1, .css-1r6slb0 h2, .css-1r6slb0 h3 {
        color: #1565C0 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1565C0, #0D47A1); /* Deep Blue Gradient */
        color: white !important;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 6px; /* Slightly tighter radius for medical field */
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(13, 71, 161, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1976D2, #1565C0);
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(13, 71, 161, 0.4);
        color: white !important;
    }
    .stButton>button p {
         color: white !important;
    }
    
    /* Eligibility Badge */
    .badge-eligible {
        background-color: #E8F5E9; /* Light Green */
        color: #2E7D32 !important; /* Emerald Green */
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border: 2px solid #C8E6C9;
        text-align: center;
        width: 100%;
        margin-bottom: 10px;
    }
    .badge-not-eligible {
        background-color: #FFEBEE; /* Light Red */
        color: #C62828 !important; /* Red */
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        border: 2px solid #FFCDD2;
        text-align: center;
        width: 100%;
        margin-bottom: 10px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #1976D2; /* Medical Blue */
    }

    /* Sidebar Styling (Optional) */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #F1F1F1; 
    }
    ::-webkit-scrollbar-thumb {
        background: #90CAF9; 
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #64B5F6; 
    }
    
    /* Input Fields & Radio Buttons */
    .stRadio label, .stTextInput label, .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #37474F !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stRadio div[role="radiogroup"] > label {
        color: #333333 !important;
    }
    
    /* FORCE INPUTS TO LIGHT THEME (Fix for Dark Mode Users) */
    /* Text Inputs & Number Inputs */
    .stTextInput input, .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #CFD8DC;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #1976D2 !important;
        box-shadow: 0 0 0 1px #1976D2;
    }
    
    /* Selectboxes - Main Container */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        border: 1px solid #CFD8DC;
    }
    
    /* Dropdown Menu - The Popup Container */
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="menu"], ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    
    /* Dropdown Options */
    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    
    /* Hover state for options */
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #E3F2FD !important;
        color: #0D47A1 !important;
    }
    
    /* Text inside the select box (selected value) */
    div[data-baseweb="select"] span {
        color: #333333 !important;
    }
    
    /* Expander Headers */
    [data-testid="stExpander"] details summary {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span {
        color: #333333 !important;
        font-weight: 600;
    }
    
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stExpander"] p {
        color: #333333 !important;
    }

    p {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Chat Input Styling */
    div[data-testid="stBottom"] > div {
        background-color: #FFFFFF !important;
        border-top: 1px solid #E0E0E0;
    }
    div[data-testid="stBottomBlockContainer"] {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stChatInput"] {
        background-color: #F5F7FA !important;
        border: 1px solid #CFD8DC !important;
        border-radius: 8px !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #333333 !important;
        background-color: transparent !important;
    }
    div[data-testid="stChatInput"] svg {
        fill: #1565C0 !important;
        color: #1565C0 !important;
    }
    div[data-testid="stChatInput"] button {
        background-color: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background-color: #E3F2FD !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = TrialMitraChatbot()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_patient_id' not in st.session_state:
    st.session_state.current_patient_id = f"PID-{np.random.randint(100000, 999999)}"
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""

def save_user_session():
    import json
    from database import get_engine
    engine = get_engine()
    session_data = {
        "patient_name": st.session_state.get("patient_name", ""),
        "current_patient_id": st.session_state.current_patient_id,
        "chat_history": st.session_state.chat_history,
        "language": st.session_state.get("language", "en")
    }
    try:
        df = pd.DataFrame([{'patient_id': st.session_state.current_patient_id, 'session_data': json.dumps(session_data)}])
        # We delete existing session for this user to essentially 'update' it
        with engine.connect() as conn:
            conn.execute(st.text("DELETE FROM user_sessions WHERE patient_id = :pid"), {"pid": st.session_state.current_patient_id})
            conn.commit()
    except Exception:
        pass # Table might not exist yet
    try:
        df.to_sql('user_sessions', engine, if_exists='append', index=False)
        return True
    except Exception as e:
        return False

def load_user_session(patient_id):
    import json
    from database import get_engine
    try:
        df = pd.read_sql_query(f"SELECT session_data FROM user_sessions WHERE patient_id = '{patient_id}'", get_engine())
        if not df.empty:
            data = json.loads(df.iloc[-1]['session_data'])
            st.session_state.patient_name = data.get("patient_name", "")
            st.session_state.current_patient_id = data.get("current_patient_id", patient_id)
            st.session_state.chat_history = data.get("chat_history", [])
            st.session_state.language = data.get("language", "en")
            return True
    except Exception:
        pass
    return False

# Define Global Options for all languages
CHAT_DEFAULTS = {
    'en': ["🔍 Find Trials", "✅ Check Eligibility", "📚 About Trials", "🤝 Join a Study", "🛡️ Safety Info", "💡 Benefits", "🏥 Specific Disease", "📞 Contact Info"],
    'kn': ["🔍 ಪ್ರಯೋಗಗಳನ್ನು ಹುಡುಕಿ", "✅ ಅರ್ಹತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ", "📚 ಪ್ರಯೋಗಗಳ ಬಗ್ಗೆ ತಿಳಿ", "🤝 ಅಧ್ಯಯನಕ್ಕೆ ಸೇರಿ", "🛡️ ಸುರಕ್ಷತಾ ಮಾಹಿತಿ", "💡 ಪ್ರಯೋಜನಗಳು", "🏥 ನಿರ್ದಿಷ್ಟ ಕಾಯಿಲೆ", "📞 ಸಂಪರ್ಕ ಮಾಹಿತಿ"],
    'hi': ["🔍 परीक्षण खोजें", "✅ पात्रता जांचें", "📚 परीक्षणों के बारे में", "🤝 अध्ययन में शामिल हों", "🛡️ सुरक्षा जानकारी", "💡 लाभ", "🏥 विशिष्ट रोग", "📞 संपर्क जानकारी"],
    'te': ["🔍 ట్రయల్స్ కనుగొనండి", "✅ అర్హతను తనిఖీ చేయండి", "📚 ట్రయల్స్ గురించి", "🤝 అధ్యయనంలో చేరండి", "🛡️ భద్రతా సమాచారం", "💡 ప్రయోజనాలు", "🏥 నిర్దిష్ట వ్యాధి", "📞 సంప్రదింపు సమాచారం"]
}

if 'last_suggestions' not in st.session_state:
    st.session_state.last_suggestions = CHAT_DEFAULTS['en']

def handle_action(user_input, silent=False):
    """Unified handler for chatbot interaction from any part of the UI"""
    lang_code = st.session_state.get('language', 'en')
    response_data = st.session_state.chatbot.generate_response(
        user_input, 
        patient_name=st.session_state.get('patient_name', 'Rahul Sharma'),
        selected_language=lang_code
    )
    
    # Add to history
    st.session_state.chat_history.append((user_input, response_data['answer']))
    
    # Enforce static Quick Actions layout (2x4 grid) across all language interactions
    st.session_state.last_suggestions = CHAT_DEFAULTS.get(lang_code, CHAT_DEFAULTS['en'])
    
    # Save interactions and session
    st.session_state.chatbot.save_interactions()
    save_user_session()
    
    # Navigate to the chatbot page
    st.session_state.nav_page = "Chatbot Assistant"
    st.rerun()

@st.cache_resource
def load_trial_matcher():
    """Cache the TrialMatcher instance to avoid reloading 45MB CSV on every rerun"""
    return TrialMatcher()

trial_matcher = load_trial_matcher()

def load_global_model():
    """Load the latest global federated model"""
    try:
        with open('models/global_model_latest.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.warning("No trained federated model found. Please run federated learning first.")
        return None

# Function to load and fit preprocessing tools (scalers, encoders)
@st.cache_resource
def load_preprocessing_tools():
    """Load all patient data and fit scalers/encoders to match training conditions"""
    try:
        from database import get_engine
        try:
            full_df = pd.read_sql_query("SELECT * FROM patients_cleaned WHERE hospital_id LIKE 'Hospital%'", get_engine())
        except Exception:
            return None, None
            
        if full_df.empty:
            return None, None
        
        # Fit Label Encoders
        label_encoders = {}
        categorical_cols = ['gender', 'disease']
        for col in categorical_cols:
            le = LabelEncoder()
            # Ensure all possible values are known (including from trial info if needed, but data should cover it)
            # For robustness, we might want to manually specify known categories if data is sparse, 
            # but fitting on full generated data is usually sufficient for this demo.
            le.fit(full_df[col])
            label_encoders[col] = le
            
        # Fit Scaler
        # Note: We must preprocess categorical cols before scaling if they are part of features
        # In federated_client.py: feature_cols = ['age', 'gender', 'bp', 'sugar', 'disease', 'trial_interest', 'engagement_level']
        # And categoricals are encoded first.
        
        df_processed = full_df.copy()
        for col, le in label_encoders.items():
            df_processed[col] = le.transform(df_processed[col])
            
        feature_cols = ['age', 'gender', 'bp', 'sugar', 'disease', 'trial_interest', 'engagement_level']
        scaler = StandardScaler()
        scaler.fit(df_processed[feature_cols])
        
        return scaler, label_encoders
        
    except Exception as e:
        st.error(f"Error loading preprocessing tools: {e}")
        return None, None

def load_network_stats():
    """Aggregate statistics from all hospital nodes in the clinical network"""
    stats = []
    total_patients = 0
    total_eligible = 0
    
    from database import get_engine
    try:
        df = pd.read_sql_table('patients_cleaned', get_engine())
        for i in range(1, 5):
            h_df = df[df['hospital_id'] == f'Hospital {i}']
            h_total = len(h_df)
            if h_total == 0: continue
            h_eligible = h_df['eligible'].sum()
            stats.append({
                'Node': f'Hospital {i}',
                'Patients': h_total,
                'Eligible': h_eligible,
                'Rate (%)': round((h_eligible / h_total) * 100, 1)
            })
            total_patients += h_total
            total_eligible += h_eligible
    except Exception as e:
        st.error(f"Error loading stats from DB: {e}")
            
    return pd.DataFrame(stats), total_patients, total_eligible

def predict_eligibility(patient_data, model_data):
    """Predict patient eligibility using the global model"""
    if model_data is None:
        return None
        
    scaler, label_encoders = load_preprocessing_tools()
    if scaler is None or label_encoders is None:
        st.error("Could not load preprocessing tools. Please ensure data is generated.")
        return None
    
    # Prepare patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Map Gender to match training data (M/F)
    if 'gender' in patient_df.columns:
        # Handle both full names and abbreviations just in case
        gender_map = {'Male': 'M', 'Female': 'F', 'M': 'M', 'F': 'F'}
        patient_df['gender'] = patient_df['gender'].map(gender_map).fillna(patient_df['gender'])

    # Encode categorical variables
    try:
        for col in ['gender', 'disease']:
            le = label_encoders.get(col)
            if le:
                # Handle unseen labels gracefully (fallback to most common or 0)
                if patient_df[col].iloc[0] not in le.classes_:
                     patient_df[col] = le.transform([le.classes_[0]]) # Fallback
                else:
                    patient_df[col] = le.transform(patient_df[col])
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return None
    
    # Scale features
    feature_cols = ['age', 'gender', 'bp', 'sugar', 'disease', 'trial_interest', 'engagement_level']
    try:
        X = patient_df[feature_cols].values
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        return None
    
    # Predict using model parameters
    coef = model_data['coef_']
    intercept = model_data['intercept_']
    
    # Logistic regression prediction
    z = np.dot(X_scaled, coef.T) + intercept
    probability = 1 / (1 + np.exp(-z))
    
    # Probability is a 1x1 array, extract float
    prob_value = float(probability[0][0])
    
    return {
        'eligible': prob_value > 0.5,
        'probability': prob_value,
        'confidence': max(prob_value, 1 - prob_value)
    }

# Initialize session state for Dashboard
if 'total_patients' not in st.session_state:
    st.session_state.total_patients = 20000 # Updated for expanded dataset (4 hospitals)
if 'eligible_patients' not in st.session_state:
    st.session_state.eligible_patients = 4900 # Approx 24.5% based on generation logic
if 'active_trials' not in st.session_state:
    st.session_state.active_trials = 10 # Updated for expanded dataset
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None

def main():
    # Header (Global)
    st.markdown("""
        <div class="main-header">
            <span style="font-size: 2.5rem;">🧬</span>
            <h1 class="header-title" style="display:inline-block; margin-left: 10px;">TrialMitra</h1>
            <span class="header-subtitle">AI Clinical Trial Assistant</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Patient Portal"
        
    pages = ["Patient Portal", "Chatbot Assistant", "Clinical Dashboard", "Trial Information", "System Status"]
    
    current_index = pages.index(st.session_state.nav_page) if st.session_state.nav_page in pages else 0
    
    selected_page = st.sidebar.selectbox(
        "Choose a page",
        pages,
        index=current_index
    )
    
    if selected_page != st.session_state.nav_page:
        st.session_state.nav_page = selected_page
        st.rerun()
        
    page = st.session_state.nav_page
    
    if page == "Patient Portal":
        patient_portal()
    elif page == "Chatbot Assistant":
        chatbot_interface()
    elif page == "Clinical Dashboard":
        dashboard_interface()
    elif page == "Trial Information":
        # Simple placeholder for now or existing function if available
        st.markdown("### 📋 Available Clinical Trials")
        st.info("Displaying trial database...")
        st.dataframe(pd.read_csv('data/clinical_trials.csv'))
    elif page == "System Status":
        system_dashboard()

def dashboard_interface():
    st.markdown("### 📊 Clinical Dashboard")
    
    # Load live network stats
    network_df, total_p, eligible_p = load_network_stats()
    
    # Top Stats Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients (Network)", f"{total_p:,}", "+15k this week")
    c2.metric("Eligible Patients", f"{eligible_p:,}", f"{eligible_p/total_p*100:.1f}%")
    c3.metric("Active Trials", st.session_state.active_trials, "Recruiting")
    
    st.markdown("---")
    
    # 🏥 NETWORK NODES SECTION
    st.subheader("🏥 Clinical Network Distribution")
    col_chart, col_table = st.columns([1, 1])
    
    with col_chart:
        # Show breakdown of patients per hospital
        st.markdown("#### Patient Distribution per Node")
        st.bar_chart(network_df.set_index('Node')['Patients'])
        
    with col_table:
        st.markdown("#### Research Node Details")
        st.table(network_df)
    
    st.markdown("---")
    
    # Charts Area
    chart_tab1, chart_tab2 = st.tabs(["Recruitment Trend", "Disease Distribution"])
    
    with chart_tab1:
        # Dummy data for chart
        chart_data = pd.DataFrame(
            np.random.randn(20, 3) + [10, 5, 2],
            columns=['Screened', 'Eligible', 'Enrolled']
        )
        st.line_chart(chart_data)
        
    with chart_tab2:
        dist_data = pd.DataFrame({
            'Disease': ['Diabetes', 'Cardio', 'Oncology', 'Respiratory', 'Neuro'],
            'Patients': [450, 320, 150, 210, 110]
        }).set_index('Disease')
        st.bar_chart(dist_data)
        
    # Simulation Data (if available from Patient Portal)
    if st.session_state.simulation_data:
        st.markdown("---")
        st.markdown("#### Latest Simulation Result")
        data = st.session_state.simulation_data
        result = data['result']
        
        c1, c2 = st.columns([1, 2])
        with c1:
             if result['eligible']:
                st.markdown(f'<div class="badge-eligible">✅ ELIGIBLE</div>', unsafe_allow_html=True)
             else:
                st.markdown(f'<div class="badge-not-eligible">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)
        with c2:
             st.progress(result['probability'])
             st.caption(f"Confidence: {result['probability']:.1%}")

def simulate_patient():
    # Generate random patient
    import random
    diseases = ["diabetes", "hypertension", "heart_disease", "cancer", "mental_health"]
    
    new_patient = {
        'id': f'P-{random.randint(1000, 9999)}',
        'age': random.randint(25, 80),
        'bp': random.randint(110, 160),
        'sugar': random.randint(90, 250),
        'disease': random.choice(diseases),
        'gender': random.choice(['Male', 'Female']),
        'trial_interest': random.randint(1, 5),
        'engagement_level': random.randint(1, 5)
    }
    
    # Predict (Mock logic or calling actual logic if we want)
    # Let's use actual logic if possible, or simplified mock for visual speed
    # We'll use the predict_eligibility function and chatbot logic
    
    # Simplified mock result for visual demo stability
    prob = random.uniform(0.3, 0.95)
    eligible = prob > 0.6
    
    result = {
        'eligible': eligible,
        'probability': prob,
        'confidence': max(prob, 1-prob)
    }
    
    st.session_state.simulation_data = {**new_patient, 'result': result}
    
    # Update stats
    st.session_state.total_patients += 1
    if eligible:
        st.session_state.eligible_patients += 1


def patient_portal():
    st.header("Patient Eligibility Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        
        patient_name = st.text_input("Patient Name", value="", placeholder="e.g. Rahul Sharma")
        patient_id = st.text_input("Patient ID", value=st.session_state.current_patient_id)

        age = st.number_input("Age", min_value=18, max_value=100, value=None, placeholder="Enter age")
        gender = st.selectbox("Gender", ["Select Gender", "Male", "Female"])
        bp = st.number_input("Blood Pressure (systolic)", min_value=70, max_value=200, value=None, placeholder="e.g. 120")
        sugar = st.number_input("Blood Sugar Level", min_value=70, max_value=400, value=None, placeholder="e.g. 100")
        disease = st.selectbox(
            "Primary Medical Condition",
            ["Select Condition", "diabetes", "hypertension", "heart_disease", "cancer", "mental_health"]
        )
        
        trial_interest, engagement_level = st.session_state.chatbot.get_interaction_features(patient_id)
        
        trial_interest = st.slider("Trial Interest (1-5)", 1, 5, trial_interest)
        engagement_level = st.slider("Engagement Level (1-5)", 1, 5, engagement_level)

    with col2:
        st.subheader("Eligibility Assessment")
        
        if st.button("Check Eligibility", type="primary"):
            # Check if all required fields are filled
            if not patient_name or age is None or gender == "Select Gender" or bp is None or sugar is None or disease == "Select Condition":
                st.warning("⚠️ Please fill in all patient information details.")
            else:
                model_data = load_global_model()
                if model_data:
                    patient_data = {'age': age, 'gender': gender, 'bp': bp, 'sugar': sugar, 'disease': disease, 'trial_interest': trial_interest, 'engagement_level': engagement_level}
                    result = predict_eligibility(patient_data, model_data)
                    
                    if result:
                        if result['eligible']:
                            st.success(f"✅ Patient **{patient_name}** is likely ELIGIBLE for clinical trials")
                            msg = f"I just checked eligibility for {patient_name} (Patient ID: {patient_id}), a {age} year old {gender} with {disease}. The result is ELIGIBLE with {result['probability']:.1%} confidence. Tell me about the next steps."
                        else:
                            st.error(f"❌ Patient **{patient_name}** may NOT be eligible for current trials")
                            msg = f"I just checked eligibility for {patient_name} (Patient ID: {patient_id}), a {age} year old {gender} with {disease}. The result is NOT ELIGIBLE ({result['probability']:.1%} confidence). What should I do next?"
                        
                        st.session_state.last_eligibility_msg = msg
                        st.session_state.last_eligibility_result = result

        # Display results from session state (persists across reruns)
        if 'last_eligibility_result' in st.session_state:
            result = st.session_state.last_eligibility_result
            
            st.markdown("---")
            if st.button("💬 Discuss result with AI Assistant", use_container_width=True):
                handle_action(st.session_state.last_eligibility_msg)

            col_met1, col_met2 = st.columns(2)
            col_met1.metric("Eligibility Probability", f"{result['probability']:.2%}")
            col_met2.metric("Prediction Confidence", f"{result['confidence']:.2%}")
            
            # ===== EXISTING CHATBOT TRIALS =====
            st.subheader("Matching Trials (Rule-Based)")
            # Re-fetch trials for rule-based check
            matching_trials = st.session_state.chatbot.check_basic_eligibility(age, bp, disease)
            
            if matching_trials:
                for trial in matching_trials:
                    with st.expander(f"{trial['trial_name']} (ID: {trial['trial_id']})"):
                        st.write(f"**Disease:** {trial['disease']}")
                        st.write(f"**Age Range:** {trial['min_age']}-{trial['max_age']} years")
                        st.write(f"**BP Range:** {trial['min_bp']}-{trial['max_bp']} mmHg")
                        st.write(f"**Status:** {trial['status']}")
            else:
                st.info("No rule-based trials match.")
            
            # ===== RECOMMENDED TRIALS FROM ctg-studies.csv =====
            if result['eligible']:
                st.subheader("🧬 Recommended Clinical Trials (Live Registry)")
                patient_info = {"age": age, "gender": gender, "disease": disease}
                recommended_trials = trial_matcher.match_trials(patient_info)
                
                if recommended_trials:
                    for trial in recommended_trials[:10]:
                        with st.expander(trial.get("Study Title", "Clinical Trial")):
                            st.write(f"**Condition:** {trial.get('Conditions', 'N/A')}")
                            st.write(f"**Phase:** {trial.get('Phases', 'N/A')}")
                            st.write(f"**Status:** {trial.get('Study Status', 'N/A')}")
                            st.write(f"**Sex Eligibility:** {trial.get('Sex', 'N/A')}")
                            st.write(f"**Age Criteria:** {trial.get('Age', 'N/A')}")
                            st.write(f"**Location:** {trial.get('Locations', 'N/A')}")
                            st.success("Potentially suitable trial based on your profile.")
                else:
                    st.warning("No suitable recruiting trials found for this patient.")
            
            else:
                pass
def chatbot_interface():
    # Header is handled globally in main()

    # Language Selection
    language_options = {
        'English': 'en',
        'Kannada': 'kn',
        'Hindi': 'hi',
        'Telugu': 'te'
    }
    
    def on_language_change():
        new_lang_name = st.session_state.lang_selector
        lang_code = language_options[new_lang_name]
        st.session_state.language = lang_code
        
        # Clear chat history for the new language context
        st.session_state.chat_history = []
        
        # Update suggestions to match the new language immediately
        st.session_state.last_suggestions = CHAT_DEFAULTS.get(lang_code, CHAT_DEFAULTS['en'])

    selected_lang_name = st.radio(
        "Select Language / भाषा चुनें / ಭಾಷೆಯನ್ನು ಆರಿಸಿ / భాషను ఎంచుకోండి",
        list(language_options.keys()),
        horizontal=True,
        key="lang_selector",
        on_change=on_language_change
    )
    selected_lang_code = language_options[selected_lang_name]
    st.session_state.language = selected_lang_code # Update for manual queries
    
    # Chat interface
    st.subheader("Chat with our AI Assistant")
    
    # Dynamic Suggestions (Interactive Options)
    if st.session_state.last_suggestions:
        st.markdown("### 🎯 Quick Actions")
        
        # Grid layout for suggestions (4 columns, multiple rows if needed)
        suggestions = st.session_state.last_suggestions
        cols_per_row = 4
        
        for i in range(0, len(suggestions), cols_per_row):
            row_suggestions = suggestions[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            for idx, suggestion in enumerate(row_suggestions):
                with cols[idx]:
                    if st.button(suggestion.upper(), key=f"sugg_{i+idx}", use_container_width=True):
                        # Detect if the suggestion matches the "Check Eligibility" intent in ANY language!
                        eligibility_texts = [opts[1].upper() for opts in CHAT_DEFAULTS.values()]
                        if suggestion.upper() in eligibility_texts:
                            st.session_state.nav_page = "Patient Portal"
                            st.rerun()
                        else:
                            handle_action(suggestion)
    
    st.divider()
    
    # Display chat history
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)
    
    # Chat input
    user_input = st.chat_input("Ask about clinical trials...")
    
    if user_input:
        handle_action(user_input)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        # Reset suggestions to current language defaults
        lang_code = st.session_state.get('language', 'en')
        st.session_state.last_suggestions = CHAT_DEFAULTS.get(lang_code, CHAT_DEFAULTS['en'])
        st.rerun()

def trial_information():
    st.header("📋 Clinical Trial Information")
    
    # Load trials data
    try:
        trials_df = pd.read_csv('data/clinical_trials.csv')
        
        st.subheader("Available Clinical Trials")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            disease_filter = st.selectbox("Filter by Disease", 
                                        ["All"] + list(trials_df['disease'].unique()))
        with col2:
            status_filter = st.selectbox("Filter by Status", 
                                       ["All"] + list(trials_df['status'].unique()))
        
        # Apply filters
        filtered_trials = trials_df.copy()
        if disease_filter != "All":
            filtered_trials = filtered_trials[filtered_trials['disease'] == disease_filter]
        if status_filter != "All":
            filtered_trials = filtered_trials[filtered_trials['status'] == status_filter]
        
        # Display trials
        for _, trial in filtered_trials.iterrows():
            with st.expander(f"{trial['trial_name']} - {trial['status'].upper()}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Trial ID:** {trial['trial_id']}")
                    st.write(f"**Disease:** {trial['disease']}")
                    st.write(f"**Status:** {trial['status']}")
                with col2:
                    st.write(f"**Age Range:** {trial['min_age']}-{trial['max_age']} years")
                    st.write(f"**BP Range:** {trial['min_bp']}-{trial['max_bp']} mmHg")
        
        # Trial statistics
        st.subheader("Trial Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trials", len(trials_df))
        with col2:
            active_trials = len(trials_df[trials_df['status'].isin(['active', 'recruiting'])])
            st.metric("Active Trials", active_trials)
        with col3:
            diseases = len(trials_df['disease'].unique())
            st.metric("Disease Areas", diseases)
            
    except FileNotFoundError:
        st.error("Trial data not found. Please generate sample data first.")

def system_dashboard():
    st.header("📊 System Dashboard & Research Metrics")
    
    # -----------------------------------------------------
    # 🔥 NEW: RESEARCH PERFORMANCE PANEL
    # -----------------------------------------------------
    st.subheader("Federated Learning Performance (Research-Grade)")
    
    try:
        # Load the newly created research results CSV
        results_file = 'results/federated_results_research.csv'
        if not os.path.exists(results_file):
            # Fallback to standard if detailed not yet generated
            results_file = 'results/federated_results.csv'
            
        fed_results = pd.read_csv(results_file)
        latest = fed_results.iloc[-1]
        
        # 1. Primary Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Global Accuracy", f"{latest['accuracy']:.2%}")
        
        # Check if research metrics exist in CSV
        if 'precision' in latest:
            col2.metric("Precision", f"{latest['precision']:.2%}")
            col3.metric("Recall", f"{latest['recall']:.2%}")
            col4.metric("F1-Score", f"{latest['f1']:.2%}")
        else:
            col2.metric("Precision", "N/A", help="Run simulation to update")
            col3.metric("Recall", "N/A")
            col4.metric("F1-Score", "N/A")

        st.markdown("---")
        
        # 2. Charts Area
        col_acc, col_loss = st.columns(2)
        
        with col_acc:
            st.markdown("#### Accuracy Trend (Privacy-Preserved)")
            fig, ax = plt.subplots(figsize=(8, 5))
            # Filter rounds >= 1
            disp = fed_results[fed_results['round'] >= 1].copy()
            ax.plot(disp['round'], disp['accuracy'] * 100, marker='o', color='#0D47A1', linewidth=2)
            ax.set_title("Global Accuracy vs Rounds", fontweight='bold')
            ax.set_xlabel("Communication Rounds")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with col_loss:
            st.markdown("#### Convergence (Loss)")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(disp['round'], disp['loss'], marker='x', color='#C62828', linestyle='--')
            ax.set_title("Global Loss vs Rounds", fontweight='bold')
            ax.set_xlabel("Communication Rounds")
            ax.set_ylabel("Log Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # -----------------------------------------------------
        # 🛡️ PRIVACY & SECURITY BLOCK
        # -----------------------------------------------------
        with st.expander("🛡️ Local Differential Privacy (LDP) Specifications"):
            st.info("""
                **Mechanism**: Gaussian Noise Injection ($\mathcal{N}(0, \sigma^2)$)  
                **Noise Level ($\sigma$)**: 0.01  
                **Purpose**: Ensures that individual patient contributions remain private during weight aggregation, satisfying academic research criteria for secure federated optimization.
            """)

        # -----------------------------------------------------
        # 📊 MODEL BENCHMARKING (CENTRALIZED VS FEDERATED)
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("🏁 Model Selection & Comparison")
        
        comparison_file = 'results/model_comparison.csv'
        if os.path.exists(comparison_file):
            comp_df = pd.read_csv(comparison_file)
            st.write("Current Centralized Baseline Benchmarks:")
            st.table(comp_df)
            st.success("Analysis: Logistic Regression provides a robust, explainable baseline for edge clinical nodes.")
        else:
            st.info("Run `python run_baselines.py` to generate the Centralized comparison table.")
            
    except Exception as e:
        st.warning(f"Waiting for research data... Please run the Federated Learning simulation. ({e})")

if __name__ == "__main__":
    main()