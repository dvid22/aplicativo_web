import os
import requests
from flask import Blueprint, redirect, url_for, session, request, current_app
from oauthlib.oauth2 import WebApplicationClient
from flask_login import login_user
from app.models import User

oauth_bp = Blueprint("oauth", __name__)
client = WebApplicationClient(os.getenv("GOOGLE_CLIENT_ID"))

# URL de configuración de endpoints de Google
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

@oauth_bp.route("/login/google")
def login_google():
    cfg = get_google_provider_cfg()
    authorization_endpoint = cfg["authorization_endpoint"]

    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=url_for("oauth.callback", _external=True),
        scope=["openid", "email", "profile"]
    )
    return redirect(request_uri)

@oauth_bp.route("/callback")
def callback():
    code = request.args.get("code")
    cfg = get_google_provider_cfg()
    token_endpoint = cfg["token_endpoint"]

    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
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
    userinfo = userinfo_response.json()

    # Extraer nombre de usuario desde email
    username = userinfo["email"].split("@")[0]
    email = userinfo["email"]

    user = User.get_by_username(username)
    if not user:
        # Crear cuenta falsa con contraseña "google_auth"
        User.create(username, email, "google_auth")
        user = User.get_by_username(username)

    login_user(user)
    return redirect(url_for("gestures.dashboard"))

