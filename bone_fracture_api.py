from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Cargar el modelo al iniciar la aplicación
model = load_model('bone_fracture_classifier.h5')

# Define las dimensiones de entrada del modelo
IMG_CLASSIFIER_SIZE = (224, 224)

# Lista de nombres de clases
CLASS_NAMES = ['fisura', 'fractura']  # Asegúrate de que coincida con el orden en el modelo

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.985

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesa la imagen para que sea compatible con el modelo."""
    image = image.resize(IMG_CLASSIFIER_SIZE)
    image = np.array(image) / 255.0  # Normaliza los valores de los píxeles
    image = np.expand_dims(image, axis=0)  # Añade una dimensión para el batch
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Verifica que el archivo esté en la solicitud
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['image']
    
    # Abre la imagen y la procesa
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_data = preprocess_image(image)
    
    # Realiza la predicción
    prediction = model.predict(input_data)[0]  # Obtener el primer resultado de predicción
    predicted_index = int(np.argmax(prediction))  # Índice de la clase con mayor probabilidad
    confidence = float(prediction[predicted_index])  # Confianza de la predicción (probabilidad)

    # Si la confianza no supera el umbral, devolver un mensaje
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "success": 0,
            "message": "La imagen no corresponde a ninguna de las clases conocidas (fisura o fractura).",
            "confidence": round(confidence * 100, 2),
            "allProbabilities": [
                {"class": CLASS_NAMES[i], "probability": round(float(prob) * 100, 2)} for i, prob in enumerate(prediction)
            ]
        })
    
    # Devuelve la predicción detallada si supera el umbral
    return jsonify({
        "success": 1,
        "predictedClass": CLASS_NAMES[predicted_index],
        "confidence": round(confidence * 100, 2),
        "allProbabilities": [
            {"class": CLASS_NAMES[i], "probability": round(float(prob) * 100, 2)} for i, prob in enumerate(prediction)
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
