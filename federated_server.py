import flwr as fl
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

np.random.seed(42)

# =========================================================
# GLOBAL EVALUATION DATA
# =========================================================

def get_evaluate_fn():

    df = pd.read_csv("data/global_test_patients_cleaned.csv")

    # Encode categorical variables
    for col in ["gender", "disease"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = [
        "age", "gender", "bp", "sugar",
        "disease", "trial_interest", "engagement_level"
    ]

    X = df[features]
    y = df["eligible"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------

    def evaluate(server_round, parameters, config):

        # ✅ SAME MODEL AS CLIENT
        model = SGDClassifier(loss="log_loss")

        model.classes_ = np.array([0, 1])
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

        y_prob = model.predict_proba(X_scaled)
        loss = log_loss(y, y_prob, normalize=True)

        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        pre = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        print(f"Round {server_round} | Global Metrics -> Acc: {acc:.4f}, Pre: {pre:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # ✅ Save Research Metrics for UI (NEW)
        new_row = pd.DataFrame([{
            "round": server_round,
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1,
            "loss": float(loss)
        }])
        
        results_file = "results/federated_results_research.csv"
        if not os.path.exists("results"):
            os.makedirs("results")
            
        if server_round == 1:
            new_row.to_csv(results_file, index=False)
        else:
            # Append if exists, otherwise write new
            if os.path.exists(results_file):
                new_row.to_csv(results_file, mode='a', header=False, index=False)
            else:
                new_row.to_csv(results_file, index=False)

        # ✅ Save Global Model Parameters for Dashboard
        model_data = {
            'coef_': parameters[0],
            'intercept_': parameters[1]
        }
        os.makedirs('models', exist_ok=True)
        with open('models/global_model_latest.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        return float(loss), {
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1
        }

    return evaluate


# =========================================================
# INITIAL GLOBAL MODEL (SAFE INITIALIZATION)
# =========================================================

def get_initial_parameters():

    n_features = 7  # number of input features

    coef = np.zeros((1, n_features))
    intercept = np.zeros(1)

    return [coef, intercept]


# =========================================================
# METRICS AGGREGATION (FOR RESEARCH REPORTING)
# =========================================================

def weighted_average(metrics):
    """Aggregate all client metrics via weighted average"""
    
    # metrics list structure: [(num_examples, {"accuracy": 0.5, ...}), ...]
    accuracies = [m[0] * m[1]["accuracy"] for m in metrics]
    precisions = [m[0] * m[1]["precision"] for m in metrics]
    recalls = [m[0] * m[1]["recall"] for m in metrics]
    f1s = [m[0] * m[1]["f1"] for m in metrics]
    examples = [m[0] for m in metrics]

    total_examples = sum(examples)
    
    agg_metrics = {
        "accuracy": sum(accuracies) / total_examples,
        "precision": sum(precisions) / total_examples,
        "recall": sum(recalls) / total_examples,
        "f1": sum(f1s) / total_examples,
    }
    
    return agg_metrics


# =========================================================
# START FEDERATED SERVER
# =========================================================

def start_server():

    strategy = fl.server.strategy.FedAvg(

        # Use all hospitals every round
        fraction_fit=1.0,
        fraction_evaluate=1.0,

        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,

        evaluate_fn=get_evaluate_fn(),
        evaluate_metrics_aggregation_fn=weighted_average, # Added advanced aggregation
        
        # Initial global model
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_initial_parameters()
        ),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()