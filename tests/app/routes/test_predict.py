import pytest
import json
from app import create_app
from flask.testing import FlaskClient

@pytest.fixture
def client():
    app = create_app()  
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict_success(client: FlaskClient) -> None:
    """
    Test the /predict endpoint with valid input data.

    Sends a POST request with valid parameters and checks that:
        - The response status code is 200.
        - The response contains a 'prediction' field.
    """
    valid_data = {
        "Material_A_Charged_Amount": [[10]],
        "Material_B_Charged_Amount": [[20]],
        "Reactor_Volume": [[30]],
        "Material_A_Final_Concentration_Previous_Batch": [[40]]
    }

    response = client.post('/predict', json=valid_data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data 


def test_predict_no_data(client: FlaskClient) -> None:
    """
    Test the /predict endpoint when no data is provided.

    Sends a POST request with an empty JSON object and checks that:
        - The response status code is 400.
        - The response contains detailed error messages indicating
          that all required fields are missing.
    """
    response = client.post('/predict', data=json.dumps({}), content_type='application/json')
    assert response.status_code == 400
    assert response.get_json() == {"error": "No data provided"}


@pytest.mark.parametrize(
    "incomplete_data,field_missing",
    [
        (
            {
                "Material_B_Charged_Amount": [[20]],
                "Reactor_Volume": [[30]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]]
            },
            "Material_A_Charged_Amount",
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Reactor_Volume": [[30]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]]
            },
            "Material_B_Charged_Amount",
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Material_B_Charged_Amount": [[20]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]]
            },
            "Reactor_Volume",
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Material_B_Charged_Amount": [[20]],
                "Reactor_Volume": [[30]],
            },
            "Material_A_Final_Concentration_Previous_Batch",
        ),
    ]
)
def test_predict_missing_required_input(client: FlaskClient, incomplete_data: dict, field_missing: str) -> None:
    """
    Test the /predict endpoint for missing required input fields.

    Uses parameterized inputs to send POST requests with incomplete data.
    Asserts that the response status code is 400 and checks that the error message 
    correctly identifies the missing required field.

    Parameters:
        incomplete_data: The JSON data to send in the request, missing one required field.
        field_missing: The name of the field that is expected to be missing in the request.
    """    
    response = client.post('/predict', json=incomplete_data)
    assert response.status_code == 400
    json_data = response.get_json()
    assert f"Missing required input: {field_missing}" in json_data["error"]
