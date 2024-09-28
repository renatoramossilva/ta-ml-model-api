from flask import Flask
from .routes import init_routes

def create_app():
    app = Flask(__name__)
    
    # Config
    app.config.from_object('app.config.Config')

    # Init Routes
    init_routes(app)

    return app
