from flask import Flask
from flask_pymongo import PyMongo
from flask_login import LoginManager
from config import Config

# Instancias de extensiones
mongo = PyMongo()
login_manager = LoginManager()
login_manager.login_view = "auth.login"  # Redirige a esta ruta si no está logueado

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Inicializar extensiones
    mongo.init_app(app)
    login_manager.init_app(app)

    # Importar modelo y definir loader de usuario
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    # Registrar blueprints
    from app.routes.auth import auth_bp
    from app.routes.gestures import gestures_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(gestures_bp)

    # Ruta principal
    @app.route("/")
    def home():
        from flask import render_template
        return render_template("index.html")

    return app
