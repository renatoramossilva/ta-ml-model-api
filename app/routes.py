from flask import current_app

def init_routes(app):
    @app.route('/')
    def home():
        return "Hello, Flask!"
