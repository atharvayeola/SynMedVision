import subprocess
import time
import requests
import sys
import base64
import os

def verify():
    print("Starting backend server...")
    # Start the server in the background
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for server to start
        print("Waiting for server to be ready...")
        for _ in range(30):
            try:
                resp = requests.get("http://localhost:8000/health")
                if resp.status_code == 200:
                    print("Server is ready!")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            print("Server failed to start.")
            return False

        # Test generation
        print("Sending generation request...")
        payload = {
            "prompt": "a histological slide of normal tissue",
            "steps": 1,  # Low steps for speed
            "guidance_scale": 7.5,
            "seed": 42
        }
        
        resp = requests.post("http://localhost:8000/api/generate", json=payload)
        if resp.status_code != 200:
            print(f"Generation failed with status {resp.status_code}: {resp.text}")
            return False
            
        data = resp.json()
        if data["status"] != "success" or not data.get("image_base64"):
            print(f"Generation failed: {data}")
            return False
            
        if "inference_time" not in data:
            print("Error: inference_time missing from response")
            return False
            
        print(f"Inference time: {data['inference_time']:.4f}s")
            
        # Save image
        img_data = base64.b64decode(data["image_base64"])
        with open("verification_result.png", "wb") as f:
            f.write(img_data)
            
        print("Verification successful! Image saved to verification_result.png")
        return True
        
    finally:
        print("Stopping server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
