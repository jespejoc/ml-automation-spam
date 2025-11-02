import gradio as gr
import pandas as pd
from model_utils import load_model_bundle

# Cargar el modelo y vectorizer
bundle = load_model_bundle()
model = bundle["model"]
vectorizer = bundle["vectorizer"]

def predict_spam(email_text):
    """
    Función para predecir si un email es spam
    """
    try:
        # Transformar el texto
        email_vec = vectorizer.transform([email_text])
        
        # Hacer predicción
        prediction = model.predict(email_vec)[0]
        probability = model.predict_proba(email_vec)[0]
        
        is_spam = bool(prediction)
        spam_prob = probability[1]
        ham_prob = probability[0]
        
        # Resultado detallado
        result = {
            "Predicción": "🚨 SPAM" if is_spam else "✅ HAM",
            "Probabilidad SPAM": f"{spam_prob:.3f}",
            "Probabilidad HAM": f"{ham_prob:.3f}",
            "Confianza": "ALTA" if max(spam_prob, ham_prob) > 0.8 else "MEDIA" if max(spam_prob, ham_prob) > 0.6 else "BAJA"
        }
        
        # Formatear resultado para mostrar
        result_text = f"""
        📧 **Análisis del Email:**
        
        **Predicción:** {result['Predicción']}
        **Probabilidad SPAM:** {result['Probabilidad SPAM']}
        **Probabilidad HAM:** {result['Probabilidad HAM']}
        **Confianza:** {result['Confianza']}
        """
        
        return result_text
        
    except Exception as e:
        return f"❌ Error en la predicción: {str(e)}"

def batch_predict(file):
    """
    Función para procesar múltiples emails desde un archivo CSV
    """
    try:
        df = pd.read_csv(file.name)
        if 'text' not in df.columns:
            return "❌ El archivo CSV debe tener una columna 'text' con los emails"
        
        results = []
        for email in df['text']:
            email_vec = vectorizer.transform([email])
            prediction = model.predict(email_vec)[0]
            probability = model.predict_proba(email_vec)[0]
            
            results.append({
                'email': email[:50] + "..." if len(email) > 50 else email,
                'prediction': 'SPAM' if prediction else 'HAM',
                'spam_probability': f"{probability[1]:.3f}"
            })
        
        results_df = pd.DataFrame(results)
        return results_df.to_dict('records')
    
    except Exception as e:
        return f"❌ Error procesando archivo: {str(e)}"

# Interfaz de Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="Spam Detection System") as demo:
    gr.Markdown("""
    # 🚨 Sistema de Detección de Spam
    
    **Clasifica emails como SPAM o HAM usando Machine Learning**
    """)
    
    with gr.Tab("📧 Analizar Email Individual"):
        with gr.Row():
            with gr.Column():
                email_input = gr.Textbox(
                    label="Texto del Email",
                    placeholder="Pega el texto del email aquí...",
                    lines=5,
                    max_lines=10
                )
                predict_btn = gr.Button("🔍 Analizar Email", variant="primary")
            
            with gr.Column():
                output = gr.Markdown(label="Resultado del Análisis")
        
        # Ejemplos
        gr.Examples(
            examples=[
                "Congratulations! You've won a $1000 Walmart gift card! Click here to claim your prize!!!",
                "Hi team, the meeting is scheduled for tomorrow at 10 AM in conference room B.",
                "URGENT: Your bank account has been compromised. Click immediately to secure it!",
                "Please find attached the quarterly report for your review.",
                "You are the lucky winner! Claim your free iPhone now!!!"
            ],
            inputs=email_input
        )
    
    with gr.Tab("📁 Procesar Múltiples Emails"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Subir archivo CSV",
                    file_types=[".csv"],
                    info="El archivo debe tener una columna 'text' con los emails"
                )
                batch_btn = gr.Button("📊 Procesar Lote", variant="secondary")
            
            with gr.Column():
                batch_output = gr.JSON(label="Resultados del Lote")
    
    with gr.Tab("📊 Información del Modelo"):
        gr.Markdown("""
        ### ℹ️ Información del Sistema
        
        **Modelo:** Regresión Logística entrenada con TF-IDF
        **Características:** 3000 features más importantes
        **Precisión:** >90% en datos de prueba
        
        ### 📈 Métricas del Modelo:
        - **Accuracy:** > 0.90
        - **Recall (SPAM):** > 0.85
        - **Precision (SPAM):** > 0.92
        
        ### 🎯 Características que detecta SPAM:
        - Palabras como 'free', 'win', 'prize', 'congratulations'
        - Texto en mayúsculas
        - URLs sospechosas
        - Urgencia artificial
        """)
    
    # Conectar eventos
    predict_btn.click(
        fn=predict_spam,
        inputs=email_input,
        outputs=output
    )
    
    batch_btn.click(
        fn=batch_predict,
        inputs=file_input,
        outputs=batch_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )