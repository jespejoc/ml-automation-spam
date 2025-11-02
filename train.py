import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_utils import save_model_bundle, load_data

def train_spam_model():
    """Entrenar modelo de detección de spam"""
    print("🔄 Cargando datos...")
    df = load_data()
    
    # Verificar que tenemos las columnas correctas
    if 'text' not in df.columns or 'spam' not in df.columns:
        raise ValueError("El dataset debe tener columnas 'text' y 'spam'")
    
    print(f"📊 Dataset cargado: {df.shape[0]} emails")
    print(f"📈 Distribución: {df['spam'].value_counts().to_dict()}")
    
    # Preparar datos
    X = df['text']
    y = df['spam']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🔧 Vectorizando texto...")
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=2
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("🤖 Entrenando modelo...")
    # Entrenar modelo
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_vec, y_train)
    
    print("📊 Evaluando modelo...")
    # Evaluación
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Accuracy del modelo: {accuracy:.4f}")
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))
    
    # Guardar modelo y vectorizer
    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "feature_names": vectorizer.get_feature_names_out()[:20]  # Top 20 features
    }
    
    save_model_bundle(bundle)
    print(f"💾 Modelo guardado en: models/model-latest.pkl")
    
    return {
        "accuracy": float(accuracy),
        "train_samples": X_train_vec.shape[0],
        "test_samples": X_test_vec.shape[0],
        "model_type": "LogisticRegression",
        "vectorizer_type": "TfidfVectorizer"
    }

if __name__ == "__main__":
    metrics = train_spam_model()
    print(f"\n🎉 Entrenamiento completado!")
    print(f"📈 Métricas finales: {metrics}")