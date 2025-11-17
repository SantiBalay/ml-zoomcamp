import requests
import pandas as pd
import re

SERVICE_URL = "http://localhost:9696"

def to_snake_case(name):
    """Convert column name to snake_case"""
    name = name.strip()
    name = re.sub(r'[^\w\s-]', '_', name)
    name = re.sub(r'[\s-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name.lower()

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{SERVICE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("Testing /predict endpoint with single prediction...")
    
    df = pd.read_csv('./data/data.csv', nrows=1)
    target_col = 'Bankrupt?'
    
    df.columns = [to_snake_case(col) for col in df.columns]
    target_col = to_snake_case(target_col)
    
    sample_features = df.drop(columns=[target_col]).iloc[0].to_dict()
    
    response = requests.post(
        f"{SERVICE_URL}/predict",
        json=sample_features,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Bankrupt: {result['bankrupt']}")
        print(f"Threshold: {result['threshold']}\n")
    else:
        print(f"Error: {response.json()}\n")
    
    return response.status_code == 200

def test_invalid_input():
    """Test error handling with invalid input"""
    print("Testing error handling with missing features...")
    
    incomplete_data = {"roa_c_before_interest_and_depreciation_before_interest": 0.37}
    
    response = requests.post(
        f"{SERVICE_URL}/predict",
        json=incomplete_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 400

if __name__ == "__main__":
    print("="*60)
    print("Testing Bankruptcy Prediction Service")
    print("="*60)
    print()
    
    try:
        health_ok = test_health()
        single_ok = test_single_prediction()
        error_ok = test_invalid_input()
        
        print("="*60)
        print("Test Summary:")
        print(f"  Health check: {'✓' if health_ok else '✗'}")
        print(f"  Single prediction: {'✓' if single_ok else '✗'}")
        print(f"  Error handling: {'✓' if error_ok else '✗'}")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the service.")
        print(f"Make sure the service is running at {SERVICE_URL}")
        print("Start it with: python predict.py")
    except Exception as e:
        print(f"ERROR: {e}")
