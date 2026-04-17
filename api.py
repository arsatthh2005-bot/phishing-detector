from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phishing_detector import PhishingDetector # Imports your existing class

app = FastAPI(title="Phishing Detector API")

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
detector = PhishingDetector()
try:
    detector.load()
except:
    print("Model not found. Run python phishing_detector.py --train first.")

class ScanRequest(BaseModel):
    text: str

@app.post("/api/scan")
async def scan(request: ScanRequest):
    # Pass the frontend's text/URL into your predict function
    result = detector.predict(request.text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)