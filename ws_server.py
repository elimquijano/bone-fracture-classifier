from flask import Flask
from flask_socketio import SocketIO, emit
import json
import asyncio
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# MODULO 2

# Cargar el modelo al iniciar la aplicación
model = load_model('bone_fracture_classifier.h5')

# Define las dimensiones de entrada del modelo
IMG_CLASSIFIER_SIZE = (224, 224)

# Lista de nombres de clases
CLASS_NAMES = ['fisura', 'fractura']  # Asegúrate de que coincida con el orden en el modelo

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.985

def preprocess_image_predict(image: Image.Image) -> np.ndarray:
    """Preprocesa la imagen para que sea compatible con el modelo."""
    image = image.resize(IMG_CLASSIFIER_SIZE)
    image = np.array(image) / 255.0  # Normaliza los valores de los píxeles
    image = np.expand_dims(image, axis=0)  # Añade una dimensión para el batch
    return image

async def predict(image_base64):
    if not image_base64:
        return {'error': 'No image provided'}

    try:
        # Decodificar Base64 a bytes
        image_data = base64.b64decode(image_base64.split(",")[1])  # Quitar el prefijo "data:image/jpeg;base64,"
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_data = preprocess_image_predict(pil_image)
        
        # Realiza la predicción
        prediction = model.predict(input_data)[0]  # Obtener el primer resultado de predicción
        predicted_index = int(np.argmax(prediction))  # Índice de la clase con mayor probabilidad
        confidence = float(prediction[predicted_index])  # Confianza de la predicción (probabilidad)
    
        # Si la confianza no supera el umbral, devolver un mensaje
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "type": "predict",
                "success": 0,
                "message": "La imagen no corresponde a ninguna de las clases conocidas (fisura o fractura).",
                "confidence": round(confidence * 100, 2),
                "allProbabilities": [
                    {"class": CLASS_NAMES[i], "probability": round(float(prob) * 100, 2)} for i, prob in enumerate(prediction)
                ]
            }
        
        # Devuelve la predicción detallada si supera el umbral
        return {
            "type": "predict",
            "success": 1,
            "predictedClass": CLASS_NAMES[predicted_index],
            "confidence": round(confidence * 100, 2),
            "allProbabilities": [
                {"class": CLASS_NAMES[i], "probability": round(float(prob) * 100, 2)} for i, prob in enumerate(prediction)
            ]
        }
    except Exception as e:
        print(f"Error en detección: {e}")
        return {"error": str(e)}

# MODULO 3

# Carga del modelo TFLite
tflite_model_path = "singlepose.tflite"  # Ajusta el nombre a tu modelo
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Asume que el modelo SinglePose Thunder usa input [1,256,256,3] int8 en [0,255]
IMG_SIZE = (256, 256)

# MoveNet SinglePose produce un output con keypoints.
# Generalmente, la forma es [1,1,17,3] (1 persona, 17 keypoints, [y,x,score]).
# Verifica en la documentación de tu modelo TFLite exacto:
# https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4
# Este modelo produce una salida [1,1,17,3].
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Redimensionar a 256x256
    image = image.resize(IMG_SIZE)
    # Convertir a array [0,255] (uint8) si el modelo es int8
    img_array = np.array(image, dtype=np.uint8)
    # Expandir a [1,256,256,3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def decode_predictions(output_data: np.ndarray):
    # Output data: [1,1,17,3]
    # Estructura: batch=1, persona=1, keypoints=17, [y, x, score]
    keypoints = output_data[0,0,:,:]  # [17,3]
    results = []
    for i, kp in enumerate(keypoints):
        y, x, score = kp
        results.append({
            "name": KEYPOINT_NAMES[i],
            "y": float(y),
            "x": float(x),
            "score": float(score)
        })
    return results

async def detection(image_base64):
    if not image_base64:
        return json.dumps({'error': 'No image provided'})

    try:
        # Decodificar Base64 a bytes
        image_data = base64.b64decode(image_base64.split(",")[1])  # Quitar el prefijo "data:image/jpeg;base64,"
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocesar imagen para el modelo
        input_data = preprocess_image(pil_image)

        # Ejecutar inferencia
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])  # [1,1,17,3]

        # Decodificar resultados
        keypoints_result = decode_predictions(output_data)
        return json.dumps({
            "type": "detection",
            "success": 1,
            "keypoints": keypoints_result
        })
    except Exception as e:
        print(f"Error en detección: {e}")
        return json.dumps({"error": str(e)})

@socketio.on('detection')
def handle_detection(data):
    image = data.get('image')
    result = asyncio.run(detection(image))
    emit('detection', json.loads(result))  # Convierte el JSON a diccionario

@socketio.on('predict')
def handle_predict(data):
    image = data.get("image")
    result = asyncio.run(predict(image))
    emit('predict', result)  # Ya es un diccionario, no necesita conversión

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8090, debug=True)