from flask import Flask, render_template
from flask_mail import Mail  # ✅ AGREGAR ESTA IMPORTACIÓN
from config import Config
from app.extensions import mongo, login_manager

# ✅ INICIALIZAR FLASK-MAIL
mail = Mail()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 🔎 Debug: confirmar que la URI de Mongo se cargó correctamente
    print("🔗 MONGO_URI cargada:", app.config.get("MONGO_URI"))

    # Inicializar extensiones
    mongo.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)  # ✅ INICIALIZAR MAIL CON LA APP

    # 🔧 Agregar la instancia de mongo al contexto de la app
    app.mongo = mongo

    # Importar y registrar Blueprints
    from app.routes.auth import auth_bp
    from app.routes.gestures import gestures_bp

    try:
        from app.oauth import oauth_bp
        app.register_blueprint(oauth_bp)
    except ImportError:
        print("⚠️ OAuth no incluido (módulo opcional)")

    app.register_blueprint(auth_bp)
    
    # ✅ CORREGIDO: Registrar el blueprint SIN url_prefix para archivos estáticos
    app.register_blueprint(gestures_bp)  # <- QUITAR url_prefix

    # Cargar el user_loader después de inicializar mongo
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    # ✅ INICIALIZAR EL EXTRACTOR DE GESTOS DESPUÉS DEL CONTEXTO
    with app.app_context():
        try:
            from app.routes.gestures import init_extractor
            init_extractor(app)
            print("✅ Extractor de gestos inicializado correctamente")
        except Exception as e:
            print(f"⚠️  Advertencia: No se pudo inicializar el extractor de gestos: {e}")
            print("ℹ️  El sistema funcionará pero sin capacidades de reconocimiento de gestos")

    @app.route("/")
    def home():
        return render_template("index.html")

    return app