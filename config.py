import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Config:
    # 🔐 Clave secreta y conexión base de datos
    SECRET_KEY = os.getenv("SECRET_KEY", "clave_predeterminada_segura")
    MONGO_URI = os.getenv("MONGO_URI")

    # 🔑 Autenticación Google OAuth (si aplica)
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

    # ✉️ Configuración de envío de correos con SendGrid
    MAIL_SERVER = "smtp.sendgrid.net"
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = "apikey"  # obligatorio para SendGrid
    MAIL_PASSWORD = os.getenv("SENDGRID_API_KEY")
    MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", "serniet22@gmail.com")

    # 🧩 Configuración general de la aplicación
    APP_NAME = os.getenv("APP_NAME", "Gestos AI")
    SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "serniet22@gmail.com")

    # 📬 Opciones de correo opcionales (mejor rendimiento en Render)
    MAIL_MAX_EMAILS = 1
    MAIL_SUPPRESS_SEND = False
