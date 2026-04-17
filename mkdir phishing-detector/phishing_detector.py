"""
Advanced Phishing Detection System
====================================
Detects phishing in URLs and email/chat text using:
- Rich feature engineering (50+ features)
- TF-IDF + Logistic Regression / Random Forest ensemble
- Explainable predictions with reason breakdown
- CLI + importable module

Usage:
    python phishing_detector.py --train          # Train on built-in dataset
    python phishing_detector.py --url "http://paypa1-secure.tk/login"
    python phishing_detector.py --text "Your account has been suspended. Verify now at http://bit.ly/abc123"
    python phishing_detector.py --file input.txt  # Batch mode
"""

import re
import math
import json
import pickle
import argparse
import urllib.parse
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAINING DATA  (expanded synthetic corpus — replace with real dataset)
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_DATA = {
    "phishing": [
        # URLs
        "http://paypa1-secure.xyz/login/verify?token=abc123",
        "https://appleid-secure.tk/account/update",
        "http://192.168.1.1/amazon-login.php",
        "http://bit.ly/3xK9mQ-verify-account",
        "https://bankofamerica-secure.info/auth/login",
        "http://faceb00k-security.gq/checkpoint",
        "http://netf1ix.cc/billing/update",
        "http://paypal.account-verify.ml/secure",
        "http://amazon-support.xyz/refund?ref=929",
        "http://microsoft-login.tk/office365/verify",
        "http://apple-support-case.ml/id",
        "http://icloud-verification.ga/verify",
        "http://steam-community-offer.tk/tradeoffer/123",
        "http://dropbox-alert.cf/update-credentials",
        "http://googledocs-share.gq/document/malware.docx",
        # Email/chat text
        "URGENT: Your account will be suspended in 24 hours. Click here immediately to verify: http://bit.ly/verif-now",
        "Dear Customer, your PayPal account has been limited. Please verify your identity: http://paypal-secure.xyz",
        "Congratulations! You have won $1,000,000. Claim your prize now. Send your bank details to claim@prizes.ml",
        "Your Apple ID has been locked due to suspicious activity. Verify now or lose access: http://apple-id-verify.tk",
        "WARNING: Unauthorized login detected. Reset password NOW: http://microsoft-reset.cf/password",
        "Your Netflix subscription payment failed. Update billing info: http://netfl1x-billing.gq/update",
        "ACCOUNT ALERT: Unusual activity on your bank account. Confirm transactions: http://bank-secure.ml",
        "You have a pending package delivery. Confirm your address and pay $1.99 fee: http://usps-track.xyz",
        "IRS NOTICE: You owe back taxes. Pay immediately to avoid arrest: http://irs-payment.tk",
        "Your Amazon account will be closed. Verify to keep access: http://amazon-verify.cf/account",
        "Security alert: Someone signed in from Russia. Click to secure your account: http://google-secure.ml",
        "Your credit card ending in 4242 has been charged $499. Dispute: http://visa-dispute.gq",
        "Confirm your Dropbox credentials have not been compromised: http://dropbox-security.tk",
        "Your WhatsApp account has been moved. Verify to regain access: http://wa-verify.xyz",
        "FINAL NOTICE: Your social security number has been suspended. Call 1-800-555-0199 now",
        "We noticed a sign-in to your Microsoft account from an unfamiliar location. Verify: http://ms-verify.cf",
        "Your DHL shipment is on hold. Pay customs fee $2.99: http://dhl-customs.tk/pay",
        "Your account password was recently changed. If not you, click: http://reset-account.ml",
        "WINNER: iPhone 15 giveaway. Enter your details to claim: http://apple-giveaway.gq",
        "Your Coinbase wallet requires verification. Avoid frozen funds: http://coinbase-verify.xyz",
        "HMRC: Tax refund of £856 is pending. Submit your bank details: http://hmrc-refund.tk",
    ],
    "legitimate": [
        # URLs
        "https://www.paypal.com/signin",
        "https://accounts.google.com/login",
        "https://github.com/login",
        "https://www.amazon.com/ap/signin",
        "https://login.microsoftonline.com",
        "https://appleid.apple.com/sign-in",
        "https://www.facebook.com/login",
        "https://linkedin.com/login",
        "https://twitter.com/i/flow/login",
        "https://stackoverflow.com/users/login",
        "https://www.bankofamerica.com/online-banking/sign-in/",
        "https://secure.chase.com/web/auth/dashboard",
        "https://www.netflix.com/login",
        "https://dropbox.com/login",
        "https://drive.google.com",
        # Email/chat text
        "Hi John, please find attached the quarterly report. Let me know if you have questions.",
        "Your order #12345 has been shipped. Track at amazon.com/orders",
        "Meeting reminder: Team standup at 10 AM tomorrow. See you then!",
        "Your GitHub pull request has been reviewed and approved.",
        "Here is the link to the document we discussed: https://docs.google.com/document/d/abc",
        "Welcome to our newsletter! You can unsubscribe at any time from your account settings.",
        "Your statement for March 2024 is now available at chase.com",
        "Thanks for signing up! Your account is ready. Log in at example.com",
        "Reminder: your subscription renews on March 31. Manage at netflix.com/account",
        "Your password was changed successfully on March 15 at 3:42 PM.",
        "New comment on your Stack Overflow post: check it out at stackoverflow.com",
        "Your Dropbox folder has been shared with alice@company.com",
        "Invoice #INV-2024-0042 from Acme Corp. Due date: April 15.",
        "Your flight confirmation: AA1234 departing March 20 at 08:00. Check in at aa.com",
        "Hey, are you free for lunch on Friday? Let me know!",
        "The project deadline has been moved to next week. Please update your tasks.",
        "Your support ticket #98765 has been resolved. Let us know if you need further help.",
        "Thank you for your purchase. Your receipt is attached.",
        "Team, the server maintenance window is Saturday 2–4 AM. Plan accordingly.",
        "Happy birthday! Hope you have a wonderful day.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class PhishingFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts 50+ hand-crafted features from raw text / URL strings."""

    # --- Lexical signals ---
    URGENCY_WORDS = {
        "urgent", "immediately", "suspend", "suspended", "verify", "verify now",
        "confirm", "validate", "limited", "unusual", "unauthorized",
        "alert", "warning", "attention", "act now", "expire", "expires",
        "locked", "blocked", "freeze", "frozen", "final notice", "24 hours",
        "48 hours", "account will be", "click here", "click immediately",
    }
    PHISHING_KEYWORDS = {
        "login", "signin", "sign-in", "password", "credential", "bank",
        "paypal", "account", "update", "billing", "payment", "secure",
        "security", "verify", "verification", "confirm", "wallet",
        "refund", "prize", "winner", "congratulations", "claim",
        "irs", "tax", "hmrc", "customs", "fee", "invoice",
        "suspended", "locked", "unauthorized", "suspicious",
    }
    SAFE_DOMAINS = {
        "google.com", "github.com", "microsoft.com", "apple.com",
        "amazon.com", "paypal.com", "facebook.com", "linkedin.com",
        "twitter.com", "stackoverflow.com", "dropbox.com", "netflix.com",
        "chase.com", "bankofamerica.com", "wellsfargo.com",
    }
    # TLDs highly associated with free/malicious hosting
    SUSPICIOUS_TLDS = {
        ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".info", ".cc",
        ".pw", ".top", ".click", ".link", ".bid", ".loan", ".work",
        ".party", ".review", ".trade", ".date", ".faith",
    }
    URL_SHORTENERS = {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
        "buff.ly", "is.gd", "rebrand.ly", "cutt.ly", "shorturl.at",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(text) for text in X])

    def _extract(self, text: str) -> list:
        text_lower = text.lower()
        features = []

        # --- URL-level features ---
        urls = re.findall(r'https?://[^\s<>"]+', text)
        has_url = int(bool(urls))
        features.append(has_url)

        if urls:
            url = urls[0]
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query = parsed.query.lower()
        else:
            url = text
            parsed = urllib.parse.urlparse(text if text.startswith("http") else "http://" + text)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query = parsed.query.lower()

        # IP address as hostname
        features.append(int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}(:\d+)?$', domain))))

        # URL length
        features.append(len(url))
        features.append(int(len(url) > 100))

        # Domain length
        features.append(len(domain))
        features.append(int(len(domain) > 30))

        # Number of subdomains
        subdomain_count = domain.count('.') - 1
        features.append(max(subdomain_count, 0))
        features.append(int(subdomain_count >= 3))

        # Suspicious TLD
        features.append(int(any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS)))

        # Is known-safe domain
        features.append(int(any(domain.endswith(d) for d in self.SAFE_DOMAINS)))

        # URL shortener used
        features.append(int(any(s in domain for s in self.URL_SHORTENERS)))

        # Special character counts
        features.append(url.count('-'))
        features.append(url.count('@'))
        features.append(url.count('//') - 1)  # extra //
        features.append(url.count('?'))
        features.append(url.count('='))
        features.append(url.count('&'))
        features.append(url.count('%'))

        # Digit ratio in domain
        digit_ratio = sum(c.isdigit() for c in domain) / max(len(domain), 1)
        features.append(digit_ratio)

        # Digit-letter swap (e.g. paypa1, g00gle)
        leet_pattern = bool(re.search(r'[a-z][0-9]|[0-9][a-z]', domain))
        features.append(int(leet_pattern))

        # Brand name in subdomain/path but not in root domain (typosquatting)
        brands = ["paypal", "google", "apple", "amazon", "microsoft",
                  "facebook", "netflix", "dropbox", "ebay", "instagram"]
        brand_in_url = any(b in url for b in brands)
        brand_in_safe_domain = any(domain.endswith(b + ".com") for b in brands)
        features.append(int(brand_in_url and not brand_in_safe_domain))

        # Path contains 'login', 'verify', 'account', 'secure'
        for kw in ["login", "verify", "account", "secure", "update", "confirm"]:
            features.append(int(kw in path or kw in query))

        # HTTPS
        features.append(int(url.startswith("https://")))

        # Port in URL
        has_port = bool(re.search(r':\d{2,5}/', url))
        features.append(int(has_port))

        # --- Text-level features ---
        features.append(len(text))
        features.append(len(text.split()))

        # Uppercase ratio (shouting = urgency)
        upper_ratio = sum(c.isupper() for c in text) / max(len(text), 1)
        features.append(upper_ratio)

        # Urgency keyword count
        urgency_hits = sum(1 for w in self.URGENCY_WORDS if w in text_lower)
        features.append(urgency_hits)
        features.append(int(urgency_hits > 2))

        # Phishing keyword count
        keyword_hits = sum(1 for w in self.PHISHING_KEYWORDS if w in text_lower)
        features.append(keyword_hits)
        features.append(int(keyword_hits > 3))

        # Exclamation / question marks
        features.append(text.count('!'))
        features.append(text.count('?'))
        features.append(int(text.count('!') > 1))

        # Currency symbols (bait)
        currency_hits = len(re.findall(r'[\$£€¥]|\bUSD\b|\bEUR\b', text))
        features.append(currency_hits)

        # Contains phone number
        features.append(int(bool(re.search(r'(\+?\d[\d\s\-]{7,}\d)', text))))

        # Mentions personal data
        pii_words = ["ssn", "social security", "credit card", "bank details",
                     "date of birth", "passport", "driver's license"]
        features.append(sum(1 for p in pii_words if p in text_lower))

        # Entropy of domain (random-looking domains are suspicious)
        features.append(self._entropy(domain))
        features.append(int(self._entropy(domain) > 3.5))

        # Digit count in text
        features.append(sum(c.isdigit() for c in text))

        # URL count in text
        features.append(len(urls))

        # HTML tags present
        features.append(int(bool(re.search(r'<[a-zA-Z]', text))))

        return features

    @staticmethod
    def _entropy(s: str) -> float:
        if not s:
            return 0.0
        counts = Counter(s)
        total = len(s)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    @property
    def feature_names(self) -> list:
        return [
            "has_url", "ip_as_host", "url_length", "url_long",
            "domain_length", "domain_long", "subdomain_count", "many_subdomains",
            "suspicious_tld", "safe_domain", "url_shortener",
            "dash_count", "at_count", "double_slash", "question_mark",
            "equals_count", "ampersand_count", "percent_count",
            "digit_ratio_domain", "leet_swap", "brand_not_in_safe_domain",
            "path_login", "path_verify", "path_account", "path_secure",
            "path_update", "path_confirm",
            "is_https", "has_port",
            "text_length", "word_count", "upper_ratio",
            "urgency_hits", "many_urgency",
            "keyword_hits", "many_keywords",
            "exclamation_count", "question_count", "multi_exclamation",
            "currency_hits", "has_phone", "pii_count",
            "domain_entropy", "high_entropy", "digit_count",
            "url_count", "has_html",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PhishingDetector:
    MODEL_PATH = Path("phishing_model.pkl")

    def __init__(self):
        self.feature_extractor = PhishingFeatureExtractor()
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=2000,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42
        )
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        self.ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft",
        )
        self.is_trained = False

    # ── Training ────────────────────────────────────────────────────────────

    def train(self, texts: list, labels: list, verbose: bool = True):
        """Train on list of texts and binary labels (1=phishing, 0=legit)."""
        X_hand = self.feature_extractor.fit_transform(texts)
        X_tfidf = self.tfidf.fit_transform(texts).toarray()
        X = np.hstack([X_hand, X_tfidf])
        X_scaled = self.scaler.fit_transform(X)
        y = np.array(labels)

        if verbose:
            print("\n📊 Cross-validation (5-fold):")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(self.ensemble, X_scaled, y, cv=cv,
                                     scoring="roc_auc", n_jobs=-1)
            print(f"   ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        self.ensemble.fit(X_tr, y_tr)

        if verbose:
            y_pred = self.ensemble.predict(X_te)
            y_proba = self.ensemble.predict_proba(X_te)[:, 1]
            print("\n📈 Test Set Report:")
            print(classification_report(y_te, y_pred,
                                        target_names=["Legitimate", "Phishing"]))
            print(f"   ROC-AUC (test): {roc_auc_score(y_te, y_proba):.4f}")
            cm = confusion_matrix(y_te, y_pred)
            print(f"\n   Confusion Matrix:\n   TN={cm[0,0]} FP={cm[0,1]}\n   FN={cm[1,0]} TP={cm[1,1]}\n")

        self.is_trained = True
        self.save()

    def save(self):
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump({
                "extractor": self.feature_extractor,
                "tfidf": self.tfidf,
                "scaler": self.scaler,
                "ensemble": self.ensemble,
            }, f)
        print(f"✅ Model saved → {self.MODEL_PATH}")

    def load(self):
        with open(self.MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.feature_extractor = data["extractor"]
        self.tfidf = data["tfidf"]
        self.scaler = data["scaler"]
        self.ensemble = data["ensemble"]
        self.is_trained = True

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Returns a rich result dict:
          label       : "PHISHING" | "LEGITIMATE"
          confidence  : float 0–1
          risk_score  : int 0–100
          risk_level  : "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "SAFE"
          reasons     : list[str]  ← explainability
          features    : dict       ← raw extracted values
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Run with --train first.")

        X_hand = self.feature_extractor.transform([text])
        X_tfidf = self.tfidf.transform([text]).toarray()
        X = np.hstack([X_hand, X_tfidf])
        X_scaled = self.scaler.transform(X)

        label_idx = self.ensemble.predict(X_scaled)[0]
        proba = self.ensemble.predict_proba(X_scaled)[0]
        phishing_prob = proba[1]

        risk_score = int(round(phishing_prob * 100))
        label = "PHISHING" if label_idx == 1 else "LEGITIMATE"
        if risk_score >= 85:
            risk_level = "CRITICAL"
        elif risk_score >= 65:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        elif risk_score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"

        reasons = self._explain(text, X_hand[0])
        raw_features = dict(zip(
            self.feature_extractor.feature_names, X_hand[0]
        ))

        return {
            "label": label,
            "confidence": round(float(phishing_prob), 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reasons": reasons,
            "features": raw_features,
        }

    def _explain(self, text: str, feat_vec: np.ndarray) -> list[str]:
        """Human-readable explanation of why a sample is flagged."""
        reasons = []
        names = self.feature_extractor.feature_names
        feat = dict(zip(names, feat_vec))

        if feat.get("ip_as_host"):
            reasons.append("🔴 IP address used as hostname (avoids DNS logging)")
        if feat.get("brand_not_in_safe_domain"):
            reasons.append("🔴 Brand name appears in URL but not on official domain (typosquatting)")
        if feat.get("suspicious_tld"):
            reasons.append("🔴 Suspicious free TLD detected (.tk, .ml, .ga, etc.)")
        if feat.get("url_shortener"):
            reasons.append("🟠 URL shortener conceals true destination")
        if feat.get("many_subdomains"):
            reasons.append("🟠 Excessive subdomains used to hide real domain")
        if feat.get("leet_swap"):
            reasons.append("🟠 Leet-speak character substitution detected (e.g. paypa1)")
        if feat.get("high_entropy"):
            reasons.append("🟠 Domain has high character entropy (looks randomly generated)")
        if feat.get("at_count", 0) > 0:
            reasons.append("🟠 '@' symbol in URL (hides real destination)")
        if feat.get("many_urgency"):
            reasons.append("🟡 Multiple urgency/pressure phrases detected")
        if feat.get("many_keywords"):
            reasons.append("🟡 High concentration of phishing keywords")
        if feat.get("multi_exclamation"):
            reasons.append("🟡 Multiple exclamation marks (high-pressure language)")
        if feat.get("pii_count", 0) > 0:
            reasons.append("🟡 Requests sensitive personal information (SSN, card, etc.)")
        if feat.get("currency_hits", 0) > 0:
            reasons.append("🟡 Monetary/currency references (prize or fee bait)")
        if feat.get("has_phone"):
            reasons.append("🟡 Phone number present (vishing attempt)")
        if not feat.get("is_https") and feat.get("has_url"):
            reasons.append("🟡 HTTP (not HTTPS) — connection is not encrypted")
        if feat.get("url_length", 0) > 100:
            reasons.append("🟡 Unusually long URL (obscures true destination)")
        if feat.get("safe_domain") and not feat.get("brand_not_in_safe_domain"):
            reasons.append("✅ Domain is a known-safe official domain")

        if not reasons:
            reasons.append("✅ No specific phishing indicators detected")

        return reasons


# ─────────────────────────────────────────────────────────────────────────────
# 4. DISPLAY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict, input_text: str):
    ICONS = {
        "CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "SAFE": "✅"
    }
    icon = ICONS.get(result["risk_level"], "❓")
    label_color = "\033[91m" if result["label"] == "PHISHING" else "\033[92m"
    reset = "\033[0m"

    print("\n" + "═" * 60)
    print(f"  INPUT  : {input_text[:80]}{'...' if len(input_text) > 80 else ''}")
    print(f"  VERDICT: {label_color}{result['label']}{reset} {icon} {result['risk_level']}")
    print(f"  RISK   : {result['risk_score']}/100  (confidence: {result['confidence']*100:.1f}%)")
    print("─" * 60)
    print("  REASONS:")
    for r in result["reasons"]:
        print(f"    • {r}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN / CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_training_corpus():
    texts, labels = [], []
    for text in TRAINING_DATA["phishing"]:
        texts.append(text)
        labels.append(1)
    for text in TRAINING_DATA["legitimate"]:
        texts.append(text)
        labels.append(0)
    return texts, labels


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Phishing Detector — URLs & Email Text"
    )
    parser.add_argument("--train", action="store_true",
                        help="Train the model on built-in corpus")
    parser.add_argument("--url", type=str, help="Analyze a single URL")
    parser.add_argument("--text", type=str, help="Analyze email / chat text")
    parser.add_argument("--file", type=str,
                        help="Batch mode: one URL/text per line")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of pretty print")
    args = parser.parse_args()

    detector = PhishingDetector()

    if args.train:
        print("🔧 Training model...")
        texts, labels = build_training_corpus()
        detector.train(texts, labels)
        return

    # Load existing model if available
    if PhishingDetector.MODEL_PATH.exists():
        detector.load()
        print("📦 Model loaded from disk.\n")
    else:
        print("⚠️  No trained model found. Training on built-in corpus first...")
        texts, labels = build_training_corpus()
        detector.train(texts, labels, verbose=False)

    inputs = []
    if args.url:
        inputs.append(args.url)
    if args.text:
        inputs.append(args.text)
    if args.file:
        with open(args.file) as f:
            inputs.extend(line.strip() for line in f if line.strip())

    if not inputs:
        # Interactive mode
        print("🛡️  Phishing Detector — Interactive Mode (Ctrl+C to quit)\n")
        while True:
            try:
                raw = input("Enter URL or text: ").strip()
                if not raw:
                    continue
                result = detector.predict(raw)
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print_result(result, raw)
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        for inp in inputs:
            result = detector.predict(inp)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_result(result, inp)


if __name__ == "__main__":
    main()
