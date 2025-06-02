from flask import Flask
from flask_pymongo import PyMongo
from flask_login import LoginManager
from config import Config

mongo = PyMongo()
login_manager = LoginManager()
login_manager.login_view = "auth.login"

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Inicializar extensiones
    mongo.init_app(app)
    login_manager.init_app(app)

    # Cargar modelos
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    # Registrar Blueprints
    from app.routes.auth import auth_bp
    from app.routes.gestures import gestures_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(gestures_bp)

    # Ruta principal
    from flask import render_template

    @app.route("/")
    def home():
        return render_template("index.html")

    return app
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
