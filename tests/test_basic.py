import pytest
from app import create_app
from flask.testing import FlaskClient


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home(client: FlaskClient) -> None:
    response = client.get("/")
    assert response.data == b"Hello, Flask!"
