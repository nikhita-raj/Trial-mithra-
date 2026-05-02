import flwr as fl
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class TrialMitraClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

        # ✅ Improved Logistic Regression via SGD
        self.model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            max_iter=1,              # keep 1 (manual epochs below)
            warm_start=True,
            class_weight='balanced',
            learning_rate='optimal',
            random_state=42
        )

        self.scaler = StandardScaler()
        self.label_encoders = {}

        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.load_and_preprocess_data()
        )

    # ---------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------

    def load_and_preprocess_data(self):
        """Load and preprocess hospital-specific patient data"""

        from database import get_engine
        engine = get_engine()
        query = f"SELECT * FROM patients_cleaned WHERE hospital_id = 'Hospital {self.hospital_id}'"
        df = pd.read_sql_query(query, engine)

        print(f"Hospital {self.hospital_id}: Loaded {len(df)} patients")

        df_processed = df.copy()

        # Encode categorical variables
        categorical_cols = ['gender', 'disease']
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le

        # Features
        feature_cols = [
            'age', 'gender', 'bp', 'sugar',
            'disease', 'trial_interest', 'engagement_level'
        ]

        X = df_processed[feature_cols]
        y = df_processed['eligible']

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Hospital {self.hospital_id}: Training samples: {len(X_train_scaled)}")
        print(f"Hospital {self.hospital_id}: Eligible ratio: {sum(y_train)/len(y_train):.2f}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    # ---------------------------------------------------------
    # FL PARAMETER HANDLING
    # ---------------------------------------------------------

    def get_parameters(self, config):
        """Return model parameters with Optional Differential Privacy (Gaussian Noise)"""

        if hasattr(self.model, "coef_"):
            params = [self.model.coef_, self.model.intercept_]
            
            # ✅ Add Gaussian Noise for Differential Privacy (Research Upgrade)
            # Standard deviation (sigma) can be adjusted for privacy/utility tradeoff
            sigma = 0.01 
            noisy_params = [
                p + np.random.normal(0, sigma, p.shape) 
                for p in params
            ]
            return noisy_params

        # ✅ Start from zeros (not random)
        n_features = self.X_train.shape[1]
        return [np.zeros((1, n_features)), np.zeros(1)]

    def set_parameters(self, parameters):
        """Set model parameters"""

        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        self.model.classes_ = np.array([0, 1])

    # ---------------------------------------------------------
    # LOCAL TRAINING (KEY FIX)
    # ---------------------------------------------------------

    def fit(self, parameters, config):
        """Train model locally"""

        self.set_parameters(parameters)

        # ⭐ MULTIPLE LOCAL EPOCHS (MAIN FIX)
        EPOCHS = 5   # you can try 5–10

        for _ in range(EPOCHS):
            X_shuffled, y_shuffled = shuffle(self.X_train, self.y_train)
            self.model.partial_fit(
                X_shuffled,
                y_shuffled,
                classes=np.array([0, 1])
            )

        # Evaluate locally
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"Hospital {self.hospital_id}: Local accuracy = {accuracy:.3f}")

        return self.get_parameters({}), len(self.X_train), {}

    # ---------------------------------------------------------
    # LOCAL EVALUATION
    # ---------------------------------------------------------

    def evaluate(self, parameters, config):
        """Evaluate model locally with full performance metrics"""

        self.set_parameters(parameters)

        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        loss = 1 - accuracy  # simple proxy loss

        print(f"Hospital {self.hospital_id}: Metrics -> Acc: {accuracy:.3f}, Pre: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.3f}")

        return loss, len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # ---------------------------------------------------------
    # NEW PATIENT PREDICTION
    # ---------------------------------------------------------

    def predict_eligibility(self, patient_data):
        """Predict eligibility for new patient"""

        patient_df = pd.DataFrame([patient_data])

        for col in ['gender', 'disease']:
            if col in self.label_encoders:
                patient_df[col] = self.label_encoders[col].transform(patient_df[col])

        feature_cols = [
            'age', 'gender', 'bp', 'sugar',
            'disease', 'trial_interest', 'engagement_level'
        ]

        X_patient = patient_df[feature_cols]
        X_patient_scaled = self.scaler.transform(X_patient)

        prediction = self.model.predict(X_patient_scaled)[0]
        probability = self.model.predict_proba(X_patient_scaled)[0]

        return {
            'eligible': bool(prediction),
            'probability': float(probability[1]),
            'confidence': float(max(probability))
        }


# ---------------------------------------------------------
# CLIENT STARTER
# ---------------------------------------------------------

def start_client(hospital_id, server_address="localhost:8080"):
    print(f"Starting Hospital {hospital_id} client...")

    client = TrialMitraClient(hospital_id)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python federated_client.py <hospital_id>")
        sys.exit(1)

    hospital_id = int(sys.argv[1])
    start_client(hospital_id)