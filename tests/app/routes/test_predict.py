import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()  
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict_success(client):
    # Example of valid input data for the model
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
def test_predict_missing_required_input(client, incomplete_data, field_missing):
    response = client.post('/predict', json=incomplete_data)
    assert response.status_code == 400
    json_data = response.get_json()
    assert f"Missing required input: {field_missing}" in json_data["error"]
