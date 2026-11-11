from pathlib import Path

from flask import Flask
from flask_cors import CORS

from medibot.routes.chat import chat_bp


def create_app():
    repository_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(repository_root),
        static_folder=str(repository_root / "static"),
    )
    CORS(app)
    app.register_blueprint(chat_bp)
    return app