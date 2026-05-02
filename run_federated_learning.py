import subprocess
import time
import os
import sys
from threading import Thread

def run_server():
    """Run the federated learning server"""
    print("Starting Federated Learning Server... (logging to server.log)")
    with open('server.log', 'w') as f:
        subprocess.run([sys.executable, "federated_server.py", "--rounds", "10", "--min-clients", "4"], stdout=f, stderr=subprocess.STDOUT)

def run_client(hospital_id, delay=0):
    """Run a hospital client"""
    if delay > 0:
        time.sleep(delay)
    
    print(f"Starting Hospital {hospital_id} Client... (logging to client{hospital_id}.log)")
    with open(f'client{hospital_id}.log', 'w') as f:
        subprocess.run([sys.executable, "federated_client.py", str(hospital_id)], stdout=f, stderr=subprocess.STDOUT)

def main():
    """Run complete federated learning simulation"""
    print("=== TrialMitra Federated Learning Simulation ===\n")
    
    # Check if data exists
    if not os.path.exists('data/hospital1_patients_cleaned.csv'):
        print("Generating sample data...")
        subprocess.run([sys.executable, "data_generator.py"])
        print("Sample data generated!\n")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Starting federated learning with 4 hospitals...")
    print("This will run 10 rounds of federated training.\n")
    
    # Run Centralized Baseline
    print("Running Centralized Baseline for comparison...")
    subprocess.run([sys.executable, "centralized_training.py", "--epochs", "10"])
    print("Centralized Baseline Completed!\n")
    
    # Start server in a separate thread
    server_thread = Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Start clients
    client1_thread = Thread(target=run_client, args=(1, 2))
    client2_thread = Thread(target=run_client, args=(2, 4))
    client3_thread = Thread(target=run_client, args=(3, 6))
    client4_thread = Thread(target=run_client, args=(4, 8))
    
    client1_thread.start()
    client2_thread.start()
    client3_thread.start()
    client4_thread.start()
    
    # Wait for clients to complete
    client1_thread.join()
    client2_thread.join()
    client3_thread.join()
    client4_thread.join()
    
    print("\n=== All Training Completed! ===")
    print("Results saved in 'results/' directory")
    print("Models saved in 'models/' directory")
    print("\nYou can now run the web interface:")
    print("streamlit run web_interface.py")

if __name__ == "__main__":
    main()