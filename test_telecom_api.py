import requests
import json

# Test the telecom upload endpoint
url = "http://localhost:5000/upload/telecom"

# Prepare the file for upload
files = {'file': open('test_telecom_correct.csv', 'rb')}

try:
    print("Testing telecom upload endpoint...")
    print(f"Uploading file: test_telecom_correct.csv")
    
    response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Session ID: {result.get('session_id')}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Exception occurred: {e}")
finally:
    files['file'].close()
