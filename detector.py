from ml_model import predict_ml

def detect_phishing(text):
    text = text.lower().strip()
    score = 0
    reasons = []

    # 🔹 Trusted safe domains
    trusted_domains = ["google.com", "amazon.in", "chatgpt.com"]

    if text.startswith("https://") and any(domain in text for domain in trusted_domains):
        return "Safe", 95, ["Trusted secure domain"]

    # 🔹 Keywords
    keywords = ["login", "verify", "bank", "urgent", "password"]
    for word in keywords:
        if word in text:
            score += 20
            reasons.append(f"Contains keyword: {word}")

    # 🔹 Email patterns
    email_patterns = [
        "dear user",
        "click below",
        "verify your account",
        "urgent action required",
        "update your details",
        "your account will be suspended"
    ]

    for pattern in email_patterns:
        if pattern in text:
            score += 30
            reasons.append(f"Email phishing pattern: {pattern}")

    # 🔹 HTTP check
    if text.startswith("http://"):
        score += 30
        reasons.append("Uses insecure HTTP")

    # 🔹 Marketing words
    if any(word in text for word in ["free", "offer", "win"]):
        score += 15
        reasons.append("Suspicious marketing words")

    # 🔹 URL structure
    if "@" in text:
        score += 25
        reasons.append("Contains @ symbol")

    if text.count('.') > 3:
        score += 15
        reasons.append("Too many subdomains")

    # 🔥 ML FEATURE BLOCK
    has_http = 1 if "http" in text else 0
    has_keyword = 1 if any(word in text for word in ["login", "verify", "bank"]) else 0
    has_urgent = 1 if "urgent" in text else 0

    ml_features = [has_http, has_keyword, has_urgent]

    ml_result = predict_ml(ml_features)

    if ml_result == 1:
        score += 30
        reasons.append("ML model detected phishing pattern")

    # 🔹 Final decision
    if score > 50:
        label = "Malicious"
    elif score > 20:
        label = "Suspicious"
    else:
        label = "Safe"

    # 🔹 Confidence
    if label == "Safe":
        confidence = 90
    else:
        confidence = min(score, 100)

    if not reasons:
        reasons.append("No suspicious patterns detected")

    return label, confidence, reasons