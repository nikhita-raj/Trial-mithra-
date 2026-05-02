import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate expanded sample datasets for the TrialMitra system"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("Generating expanded datasets...")
    
    # 1. Generate Clinical Trials Dataset (10 Trials)
    trials_data = {
        'trial_id': [f'T{i:03d}' for i in range(1, 11)],
        'trial_name': [
            'Diabetes Management Study',
            'Hypertension Control Trial',
            'Heart Disease Prevention',
            'Cancer Treatment Research',
            'Mental Health Support Study',
            'Asthma & Respiratory Care',
            'Arthritis Pain Relief',
            'Obesity & Metabolism Study',
            'Migraine Prevention Trial',
            'Kidney Function Improvement'
        ],
        'disease': [
            'diabetes', 'hypertension', 'heart_disease', 'cancer', 'mental_health',
            'respiratory', 'arthritis', 'obesity', 'migraine', 'kidney_disease'
        ],
        'min_age': [18, 25, 30, 21, 18, 18, 40, 18, 18, 30],
        'max_age': [65, 70, 75, 80, 60, 65, 85, 55, 60, 75],
        'min_bp': [80, 90, 85, 70, 75, 70, 80, 80, 70, 80],
        'max_bp': [140, 160, 150, 130, 135, 130, 150, 140, 130, 150],
        'min_sugar': [70, 70, 70, 70, 70, 70, 70, 70, 70, 70],
        'max_sugar': [180, 200, 200, 200, 200, 200, 200, 200, 200, 200], # Diabetes has stricter max (180)
        'status': ['active', 'active', 'recruiting', 'active', 'recruiting', 
                   'recruiting', 'active', 'recruiting', 'active', 'recruiting']
    }
    
    trials_df = pd.DataFrame(trials_data)
    trials_df.to_csv('data/clinical_trials.csv', index=False)
    print(f"[OK] Generated {len(trials_df)} Clinical Trials")
    
    # Indian Names Lists
    INDIAN_MALE_FIRST = ["Amit", "Arjun", "Deep", "Ishan", "Kabir", "Naveen", "Rahul", "Sahil", "Varun", "Yash", "Aditya", "Vikram", "Sanjay", "Rohan", "Karan", "Abhishek", "Manoj", "Anil", "Suresh", "Ramesh"]
    INDIAN_FEMALE_FIRST = ["Anya", "Diya", "Isha", "Kavya", "Meera", "Neha", "Pooja", "Ria", "Sneha", "Zara", "Priyanka", "Anjali", "Kiran", "Shweta", "Ritu", "Deepika", "Aditi", "Sonal", "Radhika", "Priti"]
    INDIAN_LAST_NAMES = ["Sharma", "Gupta", "Patel", "Singh", "Kumar", "Reddy", "Nair", "Iyer", "Das", "Joshi", "Choudhury", "Bose", "Mehta", "Kulkarni", "Deshmukh", "Pandey", "Verma", "Malhotra", "Khan", "Gill"]

    # Common function for patient generation
    def generate_patients(n_patients, prefix, seed, disease_probs=None):
        np.random.seed(seed)
        diseases = trials_data['disease']
        
        # Initialize lists
        ids = []
        names = []
        ages = []
        genders = []
        bps = []
        sugars = []
        patient_diseases = []
        interests = []
        engagements = []
        
        # Generate 50% eligible and 50% random/ineligible
        n_eligible = n_patients // 2
        n_random = n_patients - n_eligible
        
        # Helper for unique IDs
        def get_unique_id(existing_ids):
            while True:
                new_id = f"PID-{np.random.randint(100000, 999999)}"
                if new_id not in existing_ids:
                    return new_id

        # 1. Generate guaranteed eligible patients
        for i in range(n_eligible):
            # Pick a disease according to probabilities if provided
            d = np.random.choice(diseases, p=disease_probs)
            t_idx = trials_data['disease'].index(d)
            
            # Get valid ranges
            min_age, max_age = trials_data['min_age'][t_idx], trials_data['max_age'][t_idx]
            min_bp, max_bp = trials_data['min_bp'][t_idx], trials_data['max_bp'][t_idx]
            min_sugar, max_sugar = trials_data['min_sugar'][t_idx], trials_data['max_sugar'][t_idx]
            
            # Generate valid stats
            ages.append(np.random.randint(min_age, max_age + 1))
            bps.append(np.random.randint(min_bp, max_bp + 1))
            sugars.append(np.random.randint(min_sugar, max_sugar + 1))
            patient_diseases.append(d)
            interests.append(np.random.randint(3, 6)) # 3-5 (Eligible interest)
            
            gender_val = np.random.choice(['M', 'F'])
            genders.append(gender_val)
            
            # Select Indian Name
            if gender_val == 'M':
                first = np.random.choice(INDIAN_MALE_FIRST)
            else:
                first = np.random.choice(INDIAN_FEMALE_FIRST)
            last = np.random.choice(INDIAN_LAST_NAMES)
            names.append(f"{first} {last}")
            
            engagements.append(np.random.randint(3, 6)) # High engagement usually correlates
            ids.append(get_unique_id(ids))

        # 2. Generate random (likely ineligible) patients
        for i in range(n_random):
            d = np.random.choice(diseases, p=disease_probs)
            # Full random ranges
            ages.append(np.random.randint(18, 90))
            bps.append(np.random.randint(70, 180))
            sugars.append(np.random.randint(70, 300))
            patient_diseases.append(d)
            interests.append(np.random.randint(1, 6))
            
            gender_val = np.random.choice(['M', 'F'])
            genders.append(gender_val)
            
            # Select Indian Name
            if gender_val == 'M':
                first = np.random.choice(INDIAN_MALE_FIRST)
            else:
                first = np.random.choice(INDIAN_FEMALE_FIRST)
            last = np.random.choice(INDIAN_LAST_NAMES)
            names.append(f"{first} {last}")
            
            engagements.append(np.random.randint(1, 6))
            ids.append(get_unique_id(ids))
            
        data = {
            'patient_id': ids,
            'patient_name': names,
            'age': ages,
            'gender': genders,
            'bp': bps,
            'sugar': sugars,
            'disease': patient_diseases,
            'trial_interest': interests,
            'engagement_level': engagements
        }
        
        # Calculate eligibility (Ground Truth)
        eligible = []
        for i in range(n_patients):
            d = data['disease'][i]
            try:
                t_idx = trials_data['disease'].index(d)
                
                # Criteria
                age_ok = trials_data['min_age'][t_idx] <= data['age'][i] <= trials_data['max_age'][t_idx]
                bp_ok = trials_data['min_bp'][t_idx] <= data['bp'][i] <= trials_data['max_bp'][t_idx]
                sugar_ok = trials_data['min_sugar'][t_idx] <= data['sugar'][i] <= trials_data['max_sugar'][t_idx]
                interest_ok = data['trial_interest'][i] >= 3
                
                eligible.append(1 if (age_ok and bp_ok and sugar_ok and interest_ok) else 0)
            except:
                eligible.append(0)
                
        data['eligible'] = eligible
        df = pd.DataFrame(data)
        
        # --- Inject Data Issues for Cleaning ---
        np.random.seed(seed + 99)
        n = len(df)
        
        # 1. Missing Values (NaN in age, bp, sugar)
        # 5% missing age
        idx_age_na = np.random.choice(df.index, size=int(0.05*n), replace=False)
        df.loc[idx_age_na, 'age'] = np.nan
        # 3% missing bp
        idx_bp_na = np.random.choice(df.index, size=int(0.03*n), replace=False)
        df.loc[idx_bp_na, 'bp'] = np.nan
        # 2% missing sugar
        idx_sugar_na = np.random.choice(df.index, size=int(0.02*n), replace=False)
        df.loc[idx_sugar_na, 'sugar'] = np.nan
        
        # 2. Outliers
        # 1% unrealistic ages (e.g., 150-200)
        idx_age_outlier = np.random.choice(df.index, size=int(0.01*n), replace=False)
        df.loc[idx_age_outlier, 'age'] = np.random.randint(150, 201, size=int(0.01*n))
        # 1% unrealistic BP (e.g., 300+)
        idx_bp_outlier = np.random.choice(df.index, size=int(0.01*n), replace=False)
        df.loc[idx_bp_outlier, 'bp'] = np.random.randint(300, 400, size=int(0.01*n))
        
        # 3. Inconsistent categorical values formatting
        # Mixing 'M', 'Male', 'F', 'Female', 'female', 'male'
        m_idx = df[df['gender'] == 'M'].index
        if len(m_idx) > 0:
            idx_male = np.random.choice(m_idx, size=int(0.2*len(m_idx)), replace=False)
            df.loc[idx_male, 'gender'] = 'Male'
            idx_male_lower = np.random.choice(m_idx, size=int(0.1*len(m_idx)), replace=False)
            df.loc[idx_male_lower, 'gender'] = 'male'
            
        f_idx = df[df['gender'] == 'F'].index
        if len(f_idx) > 0:
            idx_female = np.random.choice(f_idx, size=int(0.2*len(f_idx)), replace=False)
            df.loc[idx_female, 'gender'] = 'Female'
            idx_female_lower = np.random.choice(f_idx, size=int(0.1*len(f_idx)), replace=False)
            df.loc[idx_female_lower, 'gender'] = 'female'

        # 4. Duplicates
        # Duplicate ~2% of the rows
        dupes = df.sample(frac=0.02, random_state=seed)
        df = pd.concat([df, dupes], ignore_index=True)
        
        # Shuffle to mix duplicates
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        return df

    # Generate non-IID probability distributions for diseases
    np.random.seed(42)
    alpha = np.ones(10) * 0.1 # Dirichlet parameter for higher non-IID skew (Research Upgrade)
    p_hospitals = np.random.dirichlet(alpha, size=4)
    
    # Global Test Set (Uniform distribution)
    p_global = np.ones(10) / 10.0
    
    # Engine for SQLite export
    from database import get_engine
    engine = get_engine()
    
    print("Saving to SQLite and CSV...")
    trials_df.to_sql('trials', engine, if_exists='replace', index=False)
    
    # Generate Global Test Set
    global_test_df = generate_patients(10000, 'GTP', seed=10, disease_probs=p_global)
    global_test_df.to_csv('data/global_test_patients.csv', index=False)
    global_test_df['hospital_id'] = 'GlobalTest'
    print(f"[OK] Generated {len(global_test_df)} Global Test Patients")

    # Generate Hospital 1 Dataset
    n1 = np.random.randint(20000, 35000)
    hospital1_df = generate_patients(n1, 'H1P', seed=42, disease_probs=p_hospitals[0])
    hospital1_df.to_csv('data/hospital1_patients.csv', index=False)
    hospital1_df['hospital_id'] = 'Hospital 1'
    print(f"[OK] Generated {len(hospital1_df)} Patients for Hospital 1")
    
    # Generate Hospital 2 Dataset
    n2 = np.random.randint(15000, 25000)
    hospital2_df = generate_patients(n2, 'H2P', seed=123, disease_probs=p_hospitals[1])
    hospital2_df.to_csv('data/hospital2_patients.csv', index=False)
    hospital2_df['hospital_id'] = 'Hospital 2'
    print(f"[OK] Generated {len(hospital2_df)} Patients for Hospital 2")

    # Generate Hospital 3 Dataset
    n3 = np.random.randint(35000, 45000)
    hospital3_df = generate_patients(n3, 'H3P', seed=789, disease_probs=p_hospitals[2])
    hospital3_df.to_csv('data/hospital3_patients.csv', index=False)
    hospital3_df['hospital_id'] = 'Hospital 3'
    print(f"[OK] Generated {len(hospital3_df)} Patients for Hospital 3")

    # Generate Hospital 4 Dataset
    n4 = np.random.randint(25000, 35000)
    hospital4_df = generate_patients(n4, 'H4P', seed=999, disease_probs=p_hospitals[3])
    hospital4_df.to_csv('data/hospital4_patients.csv', index=False)
    hospital4_df['hospital_id'] = 'Hospital 4'
    print(f"[OK] Generated {len(hospital4_df)} Patients for Hospital 4")

    # Save to unified SQLite 'patients' table
    all_patients_df = pd.concat([global_test_df, hospital1_df, hospital2_df, hospital3_df, hospital4_df], ignore_index=True)
    all_patients_df.to_sql('patients', engine, if_exists='replace', index=False)


    
    # 6. Generate Chatbot Interactions (10000 Interactions)
    np.random.seed(456)
    n_interactions = 10000
    
    sample_inputs = [
        "What trials are available for diabetes?",
        "Am I eligible for heart disease study?",
        "Tell me about cancer research",
        "What are the requirements?",
        "How do I join a trial?",
        "What is my eligibility status?",
        "Show me hypertension trials",
        "I want to participate in research",
        "Is there anything for migraines?",
        "Can you help me find a doctor?"
    ]
    
    intents = ['trial_inquiry', 'eligibility_check', 'general_info', 'participation_request']
    
    all_patient_ids = list(hospital1_df['patient_id']) + list(hospital2_df['patient_id']) + list(hospital3_df['patient_id']) + list(hospital4_df['patient_id'])
    
    chatbot_data = {
        'chat_id': [f'C{i:05d}' for i in range(1, n_interactions + 1)],
        'patient_id': np.random.choice(all_patient_ids, n_interactions),
        'language': np.random.choice(['english', 'hindi', 'kannada', 'telugu'], n_interactions, p=[0.6, 0.2, 0.1, 0.1]),
        'user_input': np.random.choice(sample_inputs, n_interactions),
        'intent': np.random.choice(intents, n_interactions),
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_interactions)]
    }
    
    chatbot_df = pd.DataFrame(chatbot_data)
    chatbot_df.to_csv('data/chatbot_interactions.csv', index=False)
    chatbot_df.to_sql('interactions', engine, if_exists='replace', index=False)
    print(f"[OK] Generated {len(chatbot_df)} Chatbot Interactions")
    
    print("\nData Expansion Complete!")

if __name__ == "__main__":
    generate_sample_data()