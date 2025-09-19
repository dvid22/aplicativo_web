from flask import Flask
from config import Config
from app.extensions import mongo, login_manager

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 🔎 Debug: confirmar que la URI de Mongo se cargó
    print("🔗 MONGO_URI cargada:", app.config.get("MONGO_URI"))

    # Inicializar extensiones
    mongo.init_app(app)
    login_manager.init_app(app)

    # Importar y registrar blueprints
    from app.routes.auth import auth_bp
    from app.routes.gestures import gestures_bp
    try:
        from app.oauth import oauth_bp
        app.register_blueprint(oauth_bp)
    except ImportError:
        print("⚠️ OAuth no incluido")

    app.register_blueprint(auth_bp)
    app.register_blueprint(gestures_bp)

    # Cargar user_loader después de inicializar mongo
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    @app.route("/")
    def home():
        from flask import render_template
        return render_template("index.html")

    return app
