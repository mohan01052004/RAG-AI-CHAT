import requests
import time

url = "http://127.0.0.1:8000/api/documents/upload"
pdf_path = r"D:\8th sem\Top 80 SQL Interview questions and answers.pdf"

print(f"Uploading {pdf_path} to {url}...")
start_time = time.time()

try:
    with open(pdf_path, "rb") as f:
        files = {"file": ( "Top 80 SQL Interview questions and answers.pdf", f, "application/pdf" )}
        data = {"subject": "Top 80 SQL Interview questions and answers"}
        
        # Set a 90 second timeout so we don't hang forever if there's a problem
        response = requests.post(url, files=files, data=data, timeout=90)
        
    print(f"Response Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Upload failed with error: {e}")
    print(f"Time elapsed before failure: {time.time() - start_time:.2f} seconds")
