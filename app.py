from flask import Flask, request, jsonify
from flask_cors import CORS
from detector import detect_phishing

app = Flask(__name__)
CORS(app)

# 🔹 Stats
total_scans = 0
malicious = 0
safe = 0
suspicious = 0

@app.route('/')
def home():
    return "Phishing Detection API Running"

@app.route('/check', methods=['POST'])
def check():
    global total_scans, malicious, safe, suspicious

    data = request.json

    url = data.get("url", "")
    email = data.get("email", "")

    if url:
        label, score, reasons = detect_phishing(url)
    elif email:
        label, score, reasons = detect_phishing(email)
    else:
        return jsonify({"error": "No input provided"}), 400

    total_scans += 1

    if label == "Malicious":
        malicious += 1
    elif label == "Suspicious":
        suspicious += 1
    else:
        safe += 1

    return jsonify({
        "label": label,
        "confidence": score,
        "reason": ", ".join(reasons)
    })

@app.route('/stats')
def stats():
    return jsonify({
        "total_scans": total_scans,
        "malicious": malicious,
        "suspicious": suspicious,
        "safe": safe
    })

if __name__ == '__main__':
    app.run(debug=True)s