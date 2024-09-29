"""Defines the application routes for the Flask app."""

# pylint: disable=broad-exception-caught


import numpy as np
import onnxruntime  # type: ignore
from flask import Blueprint, Flask, Response, jsonify, request
from pydantic import ValidationError
from typing import Tuple

from .schemas import PredictInput

app = Flask(__name__)

bp = Blueprint("api", __name__)

# Load the ONNX model globally
MODEL_PATH = "models/model.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH)


def init_routes(application):
    """Initialize routes for the application."""
    application.register_blueprint(bp)


@bp.route("/")
def home():
    """Return a greeting message."""
    return "Hello, Flask!"


@bp.route("/predict", methods=["POST"])
def predict() -> Tuple[Response, int]:
    """
    Handle the prediction request for the ML model.

    This endpoint accepts a POST request with JSON data containing
    the required input parameters. It processes the input data,
    runs the model inference, and returns the prediction along
    with the confidence scores.

    Request:
        JSON format with the following required fields:
        - Material_A_Charged_Amount: The amount of Material A charged.
        - Material_B_Charged_Amount: The amount of Material B charged.
        - Reactor_Volume: The volume of the reactor.
        - Material_A_Final_Concentration_Previous_Batch: The final concentration
        of Material A from the previous batch.

    Response:
        JSON format with:
        - prediction: The predicted class label.

    Returns:
        A JSON response with either the prediction and confidence scores or
        an error message if no data is provided.
    """
    validate_data = validate_input(json_data=request.get_json())
    if isinstance(validate_data, dict) and "error" in validate_data:
        return jsonify(validate_data), 400

    input_data = prepare_input_data(data=validate_data)

    # Run predict
    prediction, status_code = run_inference(input_data=input_data)
    response = {
        "prediction": prediction,
    }

    # Return predict
    return jsonify(response), status_code


def validate_input(json_data):
    """Validate the input data against the PredictInput schema."""
    try:
        return PredictInput(**json_data)
    except ValidationError as e:
        return {"error": e.errors()}
    except Exception as e:
        return {"error": str(e)}


def prepare_input_data(data):
    """Prepare input data for model inference."""
    inputs = [
        data.Material_A_Charged_Amount,
        data.Material_B_Charged_Amount,
        data.Reactor_Volume,
        data.Material_A_Final_Concentration_Previous_Batch,
    ]
    try:
        # Prepare input data to be used by the model
        print(np.array([[item[0][0]] for item in inputs], dtype=np.float32))
        return np.array([[item[0][0]] for item in inputs], dtype=np.float32)
    except Exception:
        return None


def run_inference(input_data):
    """Run inference using the ONNX model."""
    try:
        ort_inputs = {
            ort_session.get_inputs()[i].name: input_data[i : i + 1]
            for i in range(len(input_data))
        }
        output = ort_session.run(None, ort_inputs)
        return output[0].tolist(), 200  # Return prediction and no error
    except Exception as e:
        return (
            jsonify({"error": "Invalid input for the ONNX model", "details": str(e)}),
            400,
        )


if __name__ == "__main__":
    app.run(debug=True)
