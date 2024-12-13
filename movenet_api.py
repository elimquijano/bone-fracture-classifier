from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_data = preprocess_image(image)

    # Ejecutar inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # [1,1,17,3]
    
    # Decodificar keypoints
    keypoints_result = decode_predictions(output_data)

    # Se asume que SinglePose detecta a una persona principal,
    # si el score de la nariz (primer keypoint) es muy bajo, podríamos decir que no se detectó persona.
    # Aquí no hay bounding box ni múltiples personas, solo keypoints.
    # Puedes establecer un umbral si lo requieres.
    
    return jsonify({
        "success": 1,
        "keypoints": keypoints_result
    })

if __name__ == '__main__':
    # Ejecutar la API
    app.run(host='0.0.0.0', port=5010, debug=True)
