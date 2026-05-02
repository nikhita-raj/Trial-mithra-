#!/usr/bin/env python3
"""
TrialMitra - Complete Setup and Run Script
This script handles the entire TrialMitra system setup and execution
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step_num, description):
    """Print formatted step"""
    print(f"\nStep {step_num}: {description}")

def run_command(command, description="", check_success=True):
    """Run a command and handle output"""
    if description:
        print(f"   Running: {description}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if check_success and result.returncode != 0:
            print(f"   Error: {result.stderr}")
            return False
        else:
            print(f"   Success: {description}")
            return True
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_step(1, "Checking Dependencies")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'flwr', 
        'streamlit', 'flask', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   {package} is installed")
        except ImportError:
            print(f"   {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Installing missing packages: {', '.join(missing_packages)}")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        return run_command(install_cmd, "Installing dependencies")
    
    return True

def generate_data():
    """Generate sample datasets"""
    print_step(2, "Generating Sample Data")
    
    if not os.path.exists('data'):
        return run_command("python data_generator.py", "Generating datasets")
    else:
        print("   Data directory already exists")
        return True

def run_federated_learning():
    """Run federated learning training"""
    print_step(3, "Running Federated Learning")
    
    if os.path.exists('models/global_model_latest.pkl'):
        print("   Federated model already exists")
        return True
    
    print("   🔄 Starting federated learning (this may take a few minutes)...")
    return run_command("python run_federated_learning.py", "Federated learning training")

def test_chatbot():
    """Test chatbot functionality"""
    print_step(4, "Testing Chatbot")
    
    try:
        from chatbot import TrialMitraChatbot
        chatbot = TrialMitraChatbot()
        response = chatbot.generate_response("What trials are available?", patient_id="TEST001")
        print("   Chatbot is working")
        return True
    except Exception as e:
        print(f"   Chatbot error: {str(e)}")
        return False

def launch_web_interface():
    """Launch the web interface"""
    print_step(5, "Launching Web Interface")
    
    print("\n   Choose your preferred interface:")
    print("   1. Standalone HTML (Recommended - No server needed)")
    print("   2. Streamlit Web App (Full featured)")
    print("   3. Flask Web App (Custom API)")
    
    choice = input("\n   Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print("   Opening standalone HTML interface...")
        html_path = os.path.abspath("trialmitra_ui.html")
        
        # Try different methods to open the HTML file
        try:
            if sys.platform.startswith('win'):
                os.startfile(html_path)
            elif sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', html_path])
            else:  # Linux
                subprocess.run(['xdg-open', html_path])
            
            print(f"   HTML interface opened: {html_path}")
            return True
        except Exception as e:
            print(f"   Could not auto-open. Please manually open: {html_path}")
            return True
    
    elif choice == "2":
        print("   Starting Streamlit server...")
        print("   Open your browser to: http://localhost:8501")
        subprocess.run(["streamlit", "run", "web_interface.py"])
        return True
    
    elif choice == "3":
        print("   Starting Flask server...")
        print("   Open your browser to: http://localhost:5000")
        subprocess.run(["python", "flask_app.py"])
        return True
    
    else:
        print("   Invalid choice")
        return False

def show_system_status():
    """Show current system status"""
    print_step(6, "System Status Check")
    
    # Check files
    files_to_check = [
        ('data/clinical_trials.csv', 'Clinical trials data'),
        ('data/hospital1_patients_cleaned.csv', 'Hospital 1 patient data'),
        ('data/hospital2_patients_cleaned.csv', 'Hospital 2 patient data'),
        ('models/global_model_latest.pkl', 'Trained federated model'),
        ('results/federated_results.csv', 'Training results'),
        ('trialmitra_ui.html', 'Standalone web interface')
    ]
    
    print("\n   File Status:")
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"   {description}")
        else:
            print(f"   {description} - Missing")
    
    # Show statistics if available
    try:
        import pandas as pd
        
        print("\n   System Statistics:")
        
        # Hospital data
        if os.path.exists('data/hospital1_patients_cleaned.csv') and os.path.exists('data/hospital2_patients_cleaned.csv'):
            h1_df = pd.read_csv('data/hospital1_patients_cleaned.csv')
            h2_df = pd.read_csv('data/hospital2_patients_cleaned.csv')
            
            total_patients = len(h1_df) + len(h2_df)
            total_eligible = h1_df['eligible'].sum() + h2_df['eligible'].sum()
            
            print(f"   Total Patients: {total_patients}")
            print(f"   Eligible Patients: {total_eligible}")
            print(f"   Eligibility Rate: {total_eligible/total_patients*100:.1f}%")
        
        # Federated learning results
        if os.path.exists('results/federated_results.csv'):
            results_df = pd.read_csv('results/federated_results.csv')
            final_accuracy = results_df['accuracy'].iloc[-1]
            print(f"   Final Model Accuracy: {final_accuracy*100:.1f}%")
            print(f"   Training Rounds: {len(results_df)}")
        
        # Trials
        if os.path.exists('data/clinical_trials.csv'):
            trials_df = pd.read_csv('data/clinical_trials.csv')
            active_trials = len(trials_df[trials_df['status'].isin(['active', 'recruiting'])])
            print(f"   Total Trials: {len(trials_df)}")
            print(f"   Active Trials: {active_trials}")
    
    except Exception as e:
        print(f"   Could not load statistics: {str(e)}")

    return True

def main():
    """Main execution function"""
    print_header("TrialMitra - Clinical Trial Recruitment Platform")
    print("Privacy-Preserving Federated Learning System")
    
    # Step-by-step execution
    steps = [
        check_dependencies,
        generate_data,
        run_federated_learning,
        test_chatbot,
        show_system_status
    ]
    
    for step_func in steps:
        if not step_func():
            print(f"\nSetup failed at: {step_func.__name__}")
            print("Please check the error messages above and try again.")
            return False
    
    print_header("Setup Complete!")
    print("All components are ready!")
    print("Federated learning model trained!")
    print("Chatbot is functional!")
    print("Web interfaces are available!")
    
    # Launch interface
    launch_web_interface()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        print("Please check your Python environment and try again.")