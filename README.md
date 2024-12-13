# Fracture Detection and Pose Estimation with Flask and TensorFlow

This project is a Flask application that uses TensorFlow models to perform fracture detection and pose estimation. The application uses WebSockets for real-time communication between the client and the server.

## Features

- **Fracture Detection**: Uses a pre-trained TensorFlow model to classify images as either "fisura" (crack) or "fractura" (fracture).
- **Pose Estimation**: Uses a TensorFlow Lite model to detect keypoints in images, providing coordinates and scores for each keypoint.

## Requirements

To run this project, you need to install the following Python packages:

- Flask
- Flask-SocketIO
- TensorFlow
- Keras
- Pillow
- numpy
- base64
- io

You can install these packages using pip:

```bash
pip install Flask Flask-SocketIO tensorflow keras Pillow numpy base64 io
```

## Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/fracture-detection-pose-estimation.git
    cd fracture-detection-pose-estimation
    ```

2. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Place the model files**:

    - Ensure that the `bone_fracture_classifier.h5` model file is in the root directory of the project.
    - Ensure that the `singlepose.tflite` model file is in the root directory of the project.

## Running the Application

To start the Flask application, run the following command:

```bash
python app.py
```

The application will be accessible at `http://0.0.0.0:8090`.

## Endpoints

### WebSocket Endpoints

- **`/detection`**: Handles pose detection requests. Expects a JSON object with an `image` field containing a base64-encoded image.
- **`/predict`**: Handles fracture detection requests. Expects a JSON object with an `image` field containing a base64-encoded image.

## Example Usage

### Fracture Detection

Send a WebSocket message to the `/predict` endpoint with the following JSON structure:

```json
{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..."
}
```

The server will respond with a JSON object containing the prediction results:

```json
{
    "type": "predict",
    "success": 1,
    "predictedClass": "fractura",
    "confidence": 99.5,
    "allProbabilities": [
        {"class": "fisura", "probability": 0.5},
        {"class": "fractura", "probability": 99.5}
    ]
}
```

### Pose Estimation

Send a WebSocket message to the `/detection` endpoint with the following JSON structure:

```json
{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..."
}
```

The server will respond with a JSON object containing the keypoints:

```json
{
    "type": "detection",
    "success": 1,
    "keypoints": [
        {"name": "nose", "y": 123.45, "x": 67.89, "score": 0.98},
        {"name": "left_eye", "y": 120.12, "x": 70.34, "score": 0.97},
        ...
    ]
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the TensorFlow and Flask communities for their excellent documentation and support.
