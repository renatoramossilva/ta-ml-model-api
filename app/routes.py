from flask import Flask, request, jsonify
import onnxruntime
import numpy as np
from typing import Dict, Any, Tuple
 

app = Flask(__name__)

# Load the ONNX model globally
model_path = 'models/model.onnx'
ort_session = onnxruntime.InferenceSession(model_path)

def init_routes(app):
    @app.route('/')
    def home():
        return "Hello, Flask!"
    
    @app.route('/predict', methods=['POST'])
    def predict() -> Tuple[Dict[str, Any], int]:
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
        data = request.get_json()
        # Check input received
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_inputs = ['Material_A_Charged_Amount', 'Material_B_Charged_Amount', 'Reactor_Volume', 'Material_A_Final_Concentration_Previous_Batch']
        
        for input_name in required_inputs:
            if input_name not in data:
                return jsonify({"error": f"Missing required input: {input_name}"}), 400
        
        inputs = [
            data['Material_A_Charged_Amount'],
            data['Material_B_Charged_Amount'],
            data['Reactor_Volume'],
            data['Material_A_Final_Concentration_Previous_Batch']
        ]

        # Prepare input data to be used by the model
        input_data = np.array([[item[0][0]] for item in inputs], dtype=np.float32)  # Crie uma matriz 2D

        # Run predict
        ort_inputs = {ort_session.get_inputs()[i].name: input_data[i:i+1] for i in range(len(input_data))}
        output = ort_session.run(None, ort_inputs)

        # Convert ndarray to list
        prediction = output[0].tolist()  

        response = {
            "prediction": prediction,
        }

        # Return predict
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
