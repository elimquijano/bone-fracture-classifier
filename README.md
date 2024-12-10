# Bone Fracture Classifier API

This is a Flask-based API for classifying bone fractures using a pre-trained TensorFlow model. The API accepts image uploads and returns predictions indicating whether the image shows a fracture or a fissure.

## Features

- **Image Upload**: Accepts image files via POST requests.
- **Prediction**: Uses a pre-trained TensorFlow model to classify the image.
- **Confidence Threshold**: Returns detailed predictions if the confidence level exceeds a predefined threshold.

## Setup

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- Pillow
- Flask-CORS

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/bone-fracture-classifier.git
    cd bone-fracture-classifier
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Place your pre-trained model (`bone_fracture_classifier.h5`) in the root directory of the project.

### Running the API

Start the Flask application:
```sh
python app.py
```

The API will be available at `http://0.0.0.0:5010`.

## API Endpoints

### `POST /predict`

- **Description**: Upload an image to get a prediction.
- **Request**:
  - **Method**: `POST`
  - **URL**: `/predict`
  - **Body**: Form data with a key `image` containing the image file.
- **Response**:
  - **Success**:
    ```json
    {
        "success": 1,
        "predictedClass": "fractura",
        "confidence": 99.5,
        "allProbabilities": [
            {"class": "fisura", "probability": 0.5},
            {"class": "fractura", "probability": 99.5}
        ]
    }
    ```
  - **Failure**:
    ```json
    {
        "success": 0,
        "message": "La imagen no corresponde a ninguna de las clases conocidas (fisura o fractura).",
        "confidence": 45.0,
        "allProbabilities": [
            {"class": "fisura", "probability": 45.0},
            {"class": "fractura", "probability": 55.0}
        ]
    }
    ```

## Model

The model used is a pre-trained TensorFlow model (`bone_fracture_classifier.h5`). Ensure that the model file is placed in the root directory of the project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
