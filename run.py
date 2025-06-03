import os
from dotenv import load_dotenv
from flask import Flask, render_template
from config import Config

from app import mongo, login_manager

# 🔐 Permitir HTTP sin SSL solo en desarrollo local
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # ⚠️ Quita esto en producción

# Cargar variables de entorno
load_dotenv()

# Ruta a las plantillas
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'templates')

def create_app():
    print("🛠 Cargando app Flask...")
    app = Flask(__name__, template_folder=TEMPLATE_DIR)
    app.config.from_object(Config)

    # Inicializar extensiones
    mongo.init_app(app)
    login_manager.init_app(app)

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    # Blueprints
    from app.routes.auth import auth_bp
    from app.routes.gestures import gestures_bp

    try:
        from app.oauth import oauth_bp
        app.register_blueprint(oauth_bp)
    except ImportError:
        print("⚠️ OAuth no incluido")

    app.register_blueprint(auth_bp)
    app.register_blueprint(gestures_bp)

    @app.route("/")
    def home():
        return render_template("index.html")

    return app

# Ejecutar app Flask
if __name__ == "__main__":
    app = create_app()
    print("✅ App cargada, ejecutando servidor...")
    app.run(debug=True)
