import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

BUNDLE_PATH = MODELS_DIR / "model-latest.pkl"

def save_model_bundle(bundle):
    """Guardar modelo y vectorizer"""
    joblib.dump(bundle, BUNDLE_PATH)

def load_model_bundle():
    """Cargar modelo y vectorizer"""
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "models/model-latest.pkl no encontrado. "
            "Ejecuta el entrenamiento primero (train.yml)."
        )
    return joblib.load(BUNDLE_PATH)

def load_data():
    """Cargar dataset de spam"""
    try:
        df = pd.read_csv('spam.csv')
        return df
    except FileNotFoundError:
        # Crear dataset de ejemplo si no existe
        return create_sample_data()

def create_sample_data():
    """Crear dataset de ejemplo"""
    data = {
        'text': [
            "Free money now!!! Click here to claim your $1000 prize",
            "Hi John, meeting tomorrow at 3pm in conference room",
            "URGENT: Your account has been compromised",
            "Hello team, please find attached the quarterly report",
            "Congratulations! You won a free iPhone",
            "Reminder: Project deadline is next Friday",
            "Nigerian prince needs your help to transfer millions",
            "Weekly newsletter: Latest updates from our team",
            "You are a winner! Claim your prize now",
            "Hi, can we schedule a call for next week?",
            "Limited time offer: 90% off all products",
            "Your package has been delivered successfully",
            "Work from home and earn $5000 monthly",
            "Meeting notes from yesterday's discussion",
            "Your computer has viruses! Download antivirus now"
        ],
        'spam': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv('spam.csv', index=False)
    return df