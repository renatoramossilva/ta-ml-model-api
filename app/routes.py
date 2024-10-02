"""Defines the application routes for the Flask app."""

# pylint: disable=broad-exception-caught

from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnxruntime  # type: ignore
from flask import Blueprint, Flask, Response, jsonify, request
from pydantic import ValidationError

from app.logger import setup_logger

from .schemas import PredictInput

app = Flask(__name__)

bp = Blueprint("api", __name__)

# Load the ONNX model globally
MODEL_PATH = "models/model.onnx"
ort_session = onnxruntime.InferenceSession(MODEL_PATH)

LOG = setup_logger("ta-ml-model-api")


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
        LOG.error("Validation failed: %s", validate_data["error"])
        return jsonify(validate_data), 400

    input_data = prepare_input_data(data=validate_data)
    if input_data is None:
        LOG.error("Input preparation failed. Invalid data format.")
        return jsonify({"error": "Input preparation failed."}), 400

    # Run predict
    prediction, status_code = run_inference(input_data=input_data)

    if status_code == 200:
        LOG.info("Model inference successful. Prediction: %s", prediction)
    else:
        LOG.error("Model inference failed. Status code: %d", status_code)

    response = {
        "prediction": prediction,
    }

    # Return predict
    return jsonify(response), status_code


def validate_input(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the input data against the PredictInput schema.

    :param json_data: The input JSON data to validate.
            It should contain the required fields for the prediction
            as defined in the PredictInput schema.

    :return: A dictionary containing the validated input data
            if valid, or an error dictionary if validation fails.
    """
    LOG.info("Starting input validation: %s", json_data)
    try:
        return PredictInput(**json_data)
    except ValidationError as e:
        LOG.error("Validation error: %s", e.errors())
        return {"error": e.errors()}
    except Exception as e:
        LOG.error("Unexpected error during validation: %s", str(e))
        return {"error": str(e)}


def prepare_input_data(data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Prepare input data for model inference.

    This function extracts the required input parameters from the
    provided data dictionary and converts them into a NumPy array
    suitable for model inference.

    :param data: A dictionary containing the required input parameters for the model, including:
        - Material_A_Charged_Amount
        - Material_B_Charged_Amount
        - Reactor_Volume
        - Material_A_Final_Concentration_Previous_Batch

    :return:A 2D NumPy array prepared for model inference if successful,
        or None if there was an error during preparation.
    """
    inputs = [
        data.Material_A_Charged_Amount,
        data.Material_B_Charged_Amount,
        data.Reactor_Volume,
        data.Material_A_Final_Concentration_Previous_Batch,
    ]

    LOG.info("Preparing input data for model inference: %s", data)

    try:
        LOG.debug("Processing inputs: %s", inputs)

        # Prepare input data to be used by the model
        prepared_data = np.array([[item[0][0]] for item in inputs], dtype=np.float32)

        LOG.info("Input data prepared successfully: %s", prepared_data)

        return prepared_data
    except Exception as e:
        LOG.error("Error during input data preparation: %s", str(e))
        return None


def run_inference(input_data: Any) -> Tuple[list, int]:
    """
    Run inference using the ONNX model.

    This function takes input data, prepares it for the ONNX model,
    and runs the inference. It returns the model's output and the
    corresponding HTTP status code.

    :param input_data: The input data to be used for inference,
        expected to be a NumPy array or a structure compatible
        with the ONNX model.

    :return: A tuple containing the model's prediction as a
        list and the HTTP status code. The status code is 200 for
        successful inference and 400 for errors.
    """

    LOG.info("Starting model inference with input data: %s", input_data)

    try:
        ort_inputs = {
            ort_session.get_inputs()[i].name: input_data[i : i + 1]
            for i in range(len(input_data))
        }

        LOG.debug("Prepared ONNX model inputs: %s", ort_inputs)
        output = ort_session.run(None, ort_inputs)

        return output[0].tolist(), 200  # Return prediction and no error
    except Exception as e:
        LOG.error("Error during model inference: %s", str(e))
        return (
            jsonify({"error": "Invalid input for the ONNX model", "details": str(e)}),
            400,
        )


if __name__ == "__main__":
    app.run(debug=True)
