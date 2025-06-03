import os
import requests
from flask import Blueprint, redirect, request, url_for
from oauthlib.oauth2 import WebApplicationClient
from flask_login import login_user
from app.models import User

# Crear Blueprint para OAuth
oauth_bp = Blueprint("oauth", __name__)

# Inicializar cliente OAuth2
client = WebApplicationClient(os.getenv("GOOGLE_CLIENT_ID"))

# URL de configuración de endpoints de Google
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

@oauth_bp.route("/login/google")
def login_google():
    cfg = get_google_provider_cfg()
    authorization_endpoint = cfg["authorization_endpoint"]

    # ✅ URI fija que coincide exactamente con Google Cloud Console
    redirect_uri = "http://localhost:5000/callback"

    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"]
    )

    # (Opcional: imprimir para debug)
    print("🔗 REDIRECT_URI usado:", redirect_uri)

    return redirect(request_uri)

@oauth_bp.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return "No se recibió el código de autorización.", 400

    cfg = get_google_provider_cfg()
    token_endpoint = cfg["token_endpoint"]

    # Debe coincidir exactamente con el redirect_uri usado en login_google
    redirect_uri = "http://localhost:5000/callback"

    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=redirect_uri,
        code=code,
    )

    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET")),
    )
    client.parse_request_body_response(token_response.text)

    # Obtener información del usuario
    userinfo_endpoint = cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    if not userinfo_response.ok:
        return "Error al obtener información del usuario", 500

    userinfo = userinfo_response.json()
    username = userinfo["email"].split("@")[0]
    email = userinfo["email"]

    # Buscar o crear usuario
    user = User.get_by_username(username)
    if not user:
        User.create(username, email, "google_auth")
        user = User.get_by_username(username)

    login_user(user)
    return redirect(url_for("gestures.dashboard"))

