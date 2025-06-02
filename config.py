import os
from dotenv import load_dotenv

# Carga las variables desde el archivo .env
load_dotenv()

class Config:
    # Clave secreta para sesiones Flask y protección CSRF
    SECRET_KEY = os.getenv("SECRET_KEY", "clave_predeterminada_segura")

    # URI para conexión a MongoDB Atlas (incluye nombre de base de datos)
    MONGO_URI = os.getenv("MONGO_URI")

    # Configuración para autenticación con Google OAuth2
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

    # Para producción, podrías agregar:
    # SESSION_COOKIE_SECURE = True
    # REMEMBER_COOKIE_DURATION = timedelta(days=7)
