"""
Initialization module for the application.
"""

from flask import Flask
from .routes import init_routes


def create_app():
    """
    Create and configure the Flask application.

    Returns: Configured Flask application instance.
    """
    app = Flask(__name__)

    # Config
    app.config.from_object("app.config.Config")

    # Init Routes
    init_routes(app)

    return app
