import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

pdf_path = r"D:\8th sem\Top 80 SQL Interview questions and answers.pdf"
with open(pdf_path, "rb") as f:
    file_content = f.read()

print(f"PDF size: {len(file_content)} bytes")
start_time = time.time()

try:
    print("Sending entire PDF to Gemini for full OCR...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=file_content,
                mime_type='application/pdf',
            ),
            'Perform OCR on this scanned PDF and extract all text content page by page. '
            'Clearly separate pages with markers like "--- PAGE 1 ---", "--- PAGE 2 ---", etc. '
            'Extract everything verbatim.',
        ]
    )
    print(f"Success! Response received in {time.time() - start_time:.2f} seconds.")
    text_ascii = response.text.encode('ascii', errors='replace').decode('ascii')
    print("--- Sample output ---")
    print(text_ascii[:600])
    print("---------------------")
except Exception as e:
    print(f"Failed to send PDF: {e}")
