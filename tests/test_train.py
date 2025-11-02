from train import train_spam_model
from model_utils import load_model_bundle
import os

def test_training_runs():
    """Test que verifica que el entrenamiento funciona"""
    metrics = train_spam_model()
    
    # Verificar métricas básicas
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.5  # El modelo debe tener al menos 50% de accuracy
    assert metrics["train_samples"] > 0
    assert metrics["test_samples"] > 0

def test_model_saving():
    """Test que verifica que el modelo se guarda correctamente"""
    # Verificar que el archivo del modelo existe
    assert os.path.exists("models/model-latest.pkl")
    
    # Verificar que se puede cargar
    bundle = load_model_bundle()
    assert "model" in bundle
    assert "vectorizer" in bundle
    assert "accuracy" in bundle
    
    # Verificar que el modelo puede hacer predicciones
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    
    # Probar con un ejemplo
    test_text = "Free money now"
    test_vec = vectorizer.transform([test_text])
    prediction = model.predict(test_vec)
    probability = model.predict_proba(test_vec)
    
    assert prediction.shape[0] == 1
    assert probability.shape[0] == 1