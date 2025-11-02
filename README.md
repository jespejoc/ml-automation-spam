---
title: Spam Detection System
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: "4"
app_file: app.py
pinned: false
---

# Sistema de Detección de Spam 🤖

Sistema de Machine Learning para clasificación de emails en SPAM/HAM con pipeline completo de CI/CD.

## 🚀 Características

- **Modelo:** Regresión Logística con TF-IDF
- **Precisión:** >90% en datos de prueba
- **Interfaz:** Web app con Gradio
- **CI/CD:** Pipeline automático con GitHub Actions
- **Deploy:** Hugging Face Spaces

## 📊 Pipeline Automático

1. **CI:** Tests automáticos en cada push
2. **CT:** Entrenamiento automático diario
3. **CD:** Deploy automático a Hugging Face

## 🛠 Uso

```python
# Entrenar modelo manualmente
python train.py

# Ejecutar tests
pytest tests/

# Ejecutar app localmente
python app.py