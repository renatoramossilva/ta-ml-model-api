# pylint: disable=redefined-outer-name
"""
Unit tests for basic functionality of the Flask application.
"""
import pytest
from flask.testing import FlaskClient
from app import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home(client: FlaskClient) -> None:
    """Test the home route for the expected message."""
    response = client.get("/")
    assert response.data == b"Hello, Flask!"
