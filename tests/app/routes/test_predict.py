# pylint: disable=redefined-outer-name
"""
Tests for the predict route in the Flask application.
"""
import numpy as np
import pytest
from flask.testing import FlaskClient
from pydantic import BaseModel

import app.routes
from app import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class MockData(BaseModel):
    """
    Model representing input data for material processing.

    Attributes:
        Material_A_Charged_Amount: Amount of Material A charged.
        Material_B_Charged_Amount: Amount of Material B charged.
        Reactor_Volume: Volume of the reactor.
        Material_A_Final_Concentration_Previous_Batch:
            Final concentration of Material A from the previous batch.
    """

    Material_A_Charged_Amount: list
    Material_B_Charged_Amount: list
    Reactor_Volume: list
    Material_A_Final_Concentration_Previous_Batch: list


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
        "Material_A_Final_Concentration_Previous_Batch": [[40]],
    }

    response = client.post("/predict", json=valid_data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data


def test_predict_failed(client: FlaskClient) -> None:
    """
    Test the /predict endpoint with valid input data.

    Sends a POST request with valid parameters and checks that:
        - The response status code is 200.
        - The response contains a 'prediction' field.
    """
    valid_data = {
        "Material_A_Charged_Amount": [[10]],
        "Material_C_Charged_Amount": [[20]],
        "Reactor_Volume": [[30]],
        "Material_A_Final_Concentration_Previous_Batch": [[40]],
    }

    response = client.post("/predict", json=valid_data)
    assert response.status_code == 400


def test_validate_input_no_data() -> None:
    """
    Test the /predict endpoint when no data is provided.

    Sends a POST request with an empty JSON object and checks that:
        - The response status code is 400.
        - The response contains detailed error messages indicating
          that all required fields are missing.
    """
    response = app.routes.validate_input({})

    expected_errors = {
        "error": [
            {
                "type": "missing",
                "loc": ("Material_A_Charged_Amount",),
                "msg": "Field required",
                "input": {},
                "url": "https://errors.pydantic.dev/2.9/v/missing",
            },
            {
                "type": "missing",
                "loc": ("Material_B_Charged_Amount",),
                "msg": "Field required",
                "input": {},
                "url": "https://errors.pydantic.dev/2.9/v/missing",
            },
            {
                "type": "missing",
                "loc": ("Reactor_Volume",),
                "msg": "Field required",
                "input": {},
                "url": "https://errors.pydantic.dev/2.9/v/missing",
            },
            {
                "type": "missing",
                "loc": ("Material_A_Final_Concentration_Previous_Batch",),
                "msg": "Field required",
                "input": {},
                "url": "https://errors.pydantic.dev/2.9/v/missing",
            },
        ]
    }
    assert response == expected_errors


@pytest.mark.parametrize(
    "incomplete_data,field_missing",
    [
        (
            {
                "Material_B_Charged_Amount": [[20]],
                "Reactor_Volume": [[30]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]],
            },
            "Material_A_Charged_Amount",
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Reactor_Volume": [[30]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]],
            },
            "Material_B_Charged_Amount",
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Material_B_Charged_Amount": [[20]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]],
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
    ],
)
def test_validate_input_missing_required_input(
    incomplete_data: dict, field_missing: str
) -> None:
    """
    Test the /predict endpoint for missing required input fields.

    Uses parameterized inputs to send POST requests with incomplete data.
    Asserts that the response status code is 400 and checks that the error message
    correctly identifies the missing required field.

    Parameters:
        incomplete_data: The JSON data to send in the request, missing one required field.
        field_missing: The name of the field that is expected to be missing in the request.
    """
    response = app.routes.validate_input(incomplete_data)

    expected_error = [
        {
            "type": "missing",
            "loc": (field_missing,),
            "msg": "Field required",
            "input": incomplete_data,
            "url": "https://errors.pydantic.dev/2.9/v/missing",
        }
    ]

    assert expected_error == response["error"]


@pytest.mark.parametrize(
    "invalid_data,invalid_field",
    [
        (
            {
                "Material_A_Charged_Amount": "invalid_a",
                "Material_B_Charged_Amount": [[20]],
                "Reactor_Volume": [[30]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]],
            },
            ("Material_A_Charged_Amount",),
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Material_B_Charged_Amount": "invalid_b",
                "Reactor_Volume": [[30]],
                "Material_A_Final_Concentration_Previous_Batch": [[40]],
            },
            ("Material_B_Charged_Amount",),
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Material_B_Charged_Amount": [[20]],
                "Reactor_Volume": "invalid_volume",
                "Material_A_Final_Concentration_Previous_Batch": [[40]],
            },
            ("Reactor_Volume",),
        ),
        (
            {
                "Material_A_Charged_Amount": [[10]],
                "Material_B_Charged_Amount": [[20]],
                "Reactor_Volume": [[30]],
                "Material_A_Final_Concentration_Previous_Batch": "invalid_final",
            },
            ("Material_A_Final_Concentration_Previous_Batch",),
        ),
    ],
)
def test_validate_input_invalid_data(invalid_data: dict, invalid_field: str) -> None:
    """Test validate_input function with invalid data and check for the appropriate error."""
    response = app.routes.validate_input(invalid_data)

    assert "error" in response
    assert "type" in str(response["error"])
    assert response["error"][0]["loc"] == invalid_field


@pytest.mark.parametrize(
    "input_data,has_empty_value,result",
    [
        (
            MockData(
                Material_A_Charged_Amount=[[]],
                Material_B_Charged_Amount=[[20]],
                Reactor_Volume=[[30]],
                Material_A_Final_Concentration_Previous_Batch=[[40]],
            ),
            True,
            None,
        ),
        (
            MockData(
                Material_A_Charged_Amount=[[10]],
                Material_B_Charged_Amount=[[]],
                Reactor_Volume=[[30]],
                Material_A_Final_Concentration_Previous_Batch=[[40]],
            ),
            True,
            None,
        ),
        (
            MockData(
                Material_A_Charged_Amount=[[10]],
                Material_B_Charged_Amount=[[20]],
                Reactor_Volume=[[]],
                Material_A_Final_Concentration_Previous_Batch=[[40]],
            ),
            True,
            None,
        ),
        (
            MockData(
                Material_A_Charged_Amount=[[10]],
                Material_B_Charged_Amount=[[20]],
                Reactor_Volume=[[30]],
                Material_A_Final_Concentration_Previous_Batch=[[]],
            ),
            True,
            None,
        ),
    ],
)
def test_prepare_input_data_empty_value(
    input_data: dict, has_empty_value: bool, result: str
) -> None:
    """
    Test the /predict endpoint for missing required input fields.

    Uses parameterized inputs to send POST requests with incomplete data.
    Asserts that the response status code is 400 and checks that the error message
    correctly identifies the missing required field.

    Parameters:
        incomplete_data: The JSON data to send in the request, missing one required field.
        field_missing: The name of the field that is expected to be missing in the request.
    """
    response = app.routes.prepare_input_data(input_data)

    if has_empty_value:
        assert response == result


def test_prepare_input_valid_data() -> None:
    """
    Test prepare_input_data with valid MockData input.

    Verifies that the function returns the expected numpy array.
    """
    input_data = MockData(
        Material_A_Charged_Amount=[[10]],
        Material_B_Charged_Amount=[[20]],
        Reactor_Volume=[[30]],
        Material_A_Final_Concentration_Previous_Batch=[[40]],
    )

    expected_result = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)
    result = app.routes.prepare_input_data(input_data)

    assert np.array_equal(
        result, expected_result
    ), f"Expected {expected_result} but got {result}"
