from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
import re
import logging
from datetime import datetime, timedelta
from functools import wraps
import requests
from urllib.parse import urlencode
import secrets
import time
from flask_mail import Message

from app.models import User
from app import mail

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_TIME = timedelta(minutes=15)
PASSWORD_MIN_LENGTH = 8
RESET_TOKEN_EXPIRY = 3600

reset_tokens = {}

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

class AuthError(Exception):
    pass

def get_google_provider_cfg():
    try:
        response = requests.get(GOOGLE_DISCOVERY_URL)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error obteniendo configuración Google OAuth: {e}")
        return None

def get_google_config():
    try:
        return {
            'client_id': current_app.config.get('GOOGLE_CLIENT_ID', ''),
            'client_secret': current_app.config.get('GOOGLE_CLIENT_SECRET', '')
        }
    except RuntimeError:
        return {'client_id': '', 'client_secret': ''}

def is_google_configured():
    config = get_google_config()
    return bool(config['client_id'] and config['client_secret'])

def validate_username(username):
    if not username or len(username.strip()) < 3:
        raise AuthError("El nombre de usuario debe tener al menos 3 caracteres")
    
    if len(username) > 50:
        raise AuthError("El nombre de usuario no puede tener más de 50 caracteres")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise AuthError("El nombre de usuario solo puede contener letras, números, guiones y guiones bajos")
    
    return username.strip()

def validate_email(email):
    if not email:
        raise AuthError("El email es requerido")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise AuthError("Por favor ingresa un email válido")
    
    return email.strip().lower()

def validate_password(password):
    if not password or len(password) < PASSWORD_MIN_LENGTH:
        raise AuthError(f"La contraseña debe tener al menos {PASSWORD_MIN_LENGTH} caracteres")
    
    if len(password) < 12:
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        if not (has_upper and has_lower and has_digit):
            raise AuthError("La contraseña debe contener mayúsculas, minúsculas y números")
    
    return password

def handle_auth_error(error):
    logger.warning(f"Error de autenticación: {error}")
    flash(str(error), "danger")
    return None

def check_brute_force_prevention(user):
    if not user:
        return True
    
    if hasattr(user, 'login_attempts') and hasattr(user, 'locked_until'):
        if user.locked_until and user.locked_until > datetime.utcnow():
            remaining_time = user.locked_until - datetime.utcnow()
            minutes = int(remaining_time.total_seconds() / 60)
            raise AuthError(f"Cuenta temporalmente bloqueada. Intenta nuevamente en {minutes} minutos")
        
        if user.login_attempts >= MAX_LOGIN_ATTEMPTS:
            user.locked_until = datetime.utcnow() + LOCKOUT_TIME
            user.save()
            raise AuthError("Demasiados intentos fallidos. Cuenta bloqueada temporalmente")
    
    return True

def update_login_attempts(user, success):
    if not user or not hasattr(user, 'login_attempts'):
        return
    
    try:
        if success:
            user.login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
        else:
            user.login_attempts = getattr(user, 'login_attempts', 0) + 1
            
            if user.login_attempts >= MAX_LOGIN_ATTEMPTS:
                user.locked_until = datetime.utcnow() + LOCKOUT_TIME
                logger.warning(f"Cuenta {user.username} bloqueada por intentos fallidos")
        
        user.save()
    except Exception as e:
        logger.error(f"Error actualizando intentos de login: {e}")

def login_required_with_redirect(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated:
            flash("Ya tienes una sesión activa.", "info")
            return redirect(url_for("gestures_bp.dashboard"))
        return f(*args, **kwargs)
    return decorated_function

def generate_reset_token(user_id):
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + RESET_TOKEN_EXPIRY
    
    reset_tokens[token] = {
        'user_id': user_id,
        'expires_at': expires_at,
        'used': False
    }
    
    return token

def verify_reset_token(token):
    if token not in reset_tokens:
        return None
    
    token_data = reset_tokens[token]
    
    if time.time() > token_data['expires_at']:
        del reset_tokens[token]
        return None
    
    if token_data['used']:
        return None
    
    return token_data

def mark_token_used(token):
    if token in reset_tokens:
        reset_tokens[token]['used'] = True

def cleanup_expired_tokens():
    current_time = time.time()
    expired_tokens = [
        token for token, data in reset_tokens.items()
        if current_time > data['expires_at']
    ]
    
    for token in expired_tokens:
        del reset_tokens[token]
    
    if expired_tokens:
        logger.info(f"Limpiados {len(expired_tokens)} tokens expirados")

def send_password_reset_email(user, token):
    try:
        reset_url = url_for('auth.reset_password_confirm', token=token, _external=True)
        app_name = current_app.config.get('APP_NAME', 'Gestos AI')
        support_email = current_app.config.get('SUPPORT_EMAIL', 'serniet22@gmail.com')
        
        msg = Message(
            subject=f'Restablece tu contraseña - {app_name}',
            recipients=[user.email],
            sender=(app_name, current_app.config.get('MAIL_DEFAULT_SENDER')),
            reply_to=support_email
        )
        
        msg.html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; }}
                .header {{ background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 30px; text-align: center; }}
                .content {{ padding: 30px; background: #f8fafc; }}
                .button {{ display: inline-block; padding: 12px 30px; background: #10b981; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ padding: 20px; text-align: center; color: #64748b; font-size: 14px; }}
                .warning {{ background: #fef3c7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{app_name}</h1>
                <h2>Restablecer Contraseña</h2>
            </div>
            
            <div class="content">
                <p>Hola <strong>{user.name or user.username}</strong>,</p>
                
                <p>Has solicitado restablecer tu contraseña en {app_name}.</p>
                
                <p>Para crear una nueva contraseña, haz clic en el siguiente botón:</p>
                
                <div style="text-align: center;">
                    <a href="{reset_url}" class="button" style="color: white;">
                        Restablecer Contraseña
                    </a>
                </div>
                
                <p>O copia y pega este enlace en tu navegador:</p>
                <p style="word-break: break-all; color: #475569; background: #f1f5f9; padding: 10px; border-radius: 5px;">
                    {reset_url}
                </p>
                
                <div class="warning">
                    <p><strong>⚠️ Este enlace expirará en 1 hora.</strong></p>
                </div>
                
                <p>Si no solicitaste este cambio, puedes ignorar este mensaje. Tu cuenta permanecerá segura.</p>
            </div>
            
            <div class="footer">
                <p>Saludos,<br>El equipo de {app_name}</p>
                <p>¿Necesitas ayuda? Contáctanos: <a href="mailto:{support_email}">{support_email}</a></p>
                <p>© 2025 {app_name}. Todos los derechos reservados.</p>
            </div>
        </body>
        </html>
        """
        
        msg.body = f"""
        Hola {user.name or user.username},
        
        Has solicitado restablecer tu contraseña en {app_name}.
        
        Para crear una nueva contraseña, visita el siguiente enlace:
        {reset_url}
        
        ⚠️ Este enlace expirará en 1 hora.
        
        Si no solicitaste este cambio, puedes ignorar este mensaje.
        
        Saludos,
        El equipo de {app_name}
        
        ¿Necesitas ayuda? Contáctanos: {support_email}
        """
        
        mail.send(msg)
        logger.info(f"Email de recuperación enviado a: {user.email}")
        return True
        
    except Exception as e:
        logger.error(f"Error enviando email de recuperación: {str(e)}")
        return False

@auth_bp.route("/login", methods=["GET", "POST"])
@login_required_with_redirect
def login():
    try:
        if request.method == "POST":
            if not request.form.get("username") or not request.form.get("password"):
                flash("Por favor completa todos los campos requeridos.", "warning")
                return render_template("auth/login.html", google_enabled=is_google_configured())
            
            username = request.form["username"].strip()
            password = request.form["password"]
            remember = bool(request.form.get("remember"))
            
            user = User.get_by_username(username)
            
            try:
                check_brute_force_prevention(user)
            except AuthError as e:
                return render_template("auth/login.html", username=username, error={"general": str(e)}, google_enabled=is_google_configured())
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                update_login_attempts(user, True)
                
                logger.info(f"Login exitoso para usuario: {username}")
                flash("¡Bienvenido de nuevo! Sesión iniciada correctamente.", "success")
                
                next_page = request.args.get('next')
                if next_page and next_page.startswith('/'):
                    return redirect(next_page)
                return redirect(url_for("gestures_bp.dashboard"))
            else:
                update_login_attempts(user, False)
                remaining_attempts = MAX_LOGIN_ATTEMPTS - getattr(user, 'login_attempts', 1) if user else MAX_LOGIN_ATTEMPTS - 1
                
                error_msg = "Usuario o contraseña incorrectos."
                if remaining_attempts <= 3:
                    error_msg += f" Te quedan {remaining_attempts} intentos."
                
                logger.warning(f"Intento de login fallido para usuario: {username}")
                return render_template("auth/login.html", username=username, error={"general": error_msg}, google_enabled=is_google_configured())
        
        return render_template("auth/login.html", google_enabled=is_google_configured())
        
    except Exception as e:
        logger.error(f"Error en login: {str(e)}")
        flash("Error interno del sistema. Por favor intenta más tarde.", "danger")
        return render_template("auth/login.html", google_enabled=is_google_configured())

@auth_bp.route("/register", methods=["GET", "POST"])
@login_required_with_redirect
def register():
    try:
        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            terms = request.form.get("terms")
            
            errors = {}
            
            if not full_name or len(full_name.strip()) < 2:
                errors["full_name"] = "El nombre completo debe tener al menos 2 caracteres"
            
            try:
                username = validate_username(username)
            except AuthError as e:
                errors["username"] = str(e)
            
            try:
                email = validate_email(email)
            except AuthError as e:
                errors["email"] = str(e)
            
            try:
                password = validate_password(password)
            except AuthError as e:
                errors["password"] = str(e)
            
            if password != confirm_password:
                errors["confirm_password"] = "Las contraseñas no coinciden"
            
            if not terms:
                errors["terms"] = "Debes aceptar los términos y condiciones"
            
            if not errors.get("username") and User.get_by_username(username):
                errors["username"] = "Este nombre de usuario ya está en uso"
            
            if not errors.get("email") and User.get_by_email(email):
                errors["email"] = "Este email ya está registrado"
            
            if errors:
                return render_template("auth/register.html", full_name=full_name, username=username, email=email, error=errors, google_enabled=is_google_configured())
            
            try:
                user = User.create(username, email, password)
                if hasattr(user, 'name'):
                    user.name = full_name
                    user.save()
                logger.info(f"Nuevo usuario registrado: {username} ({email})")
                
                login_user(user)
                flash("¡Cuenta creada exitosamente! Bienvenido a Gestos AI.", "success")
                return redirect(url_for("gestures_bp.dashboard"))
                
            except Exception as e:
                logger.error(f"Error creando usuario: {str(e)}")
                errors["general"] = "Error al crear la cuenta. Por favor intenta nuevamente."
                return render_template("auth/register.html", full_name=full_name, username=username, email=email, error=errors, google_enabled=is_google_configured())
        
        return render_template("auth/register.html", google_enabled=is_google_configured())
        
    except Exception as e:
        logger.error(f"Error en registro: {str(e)}")
        flash("Error interno del sistema. Por favor intenta más tarde.", "danger")
        return render_template("auth/register.html", google_enabled=is_google_configured())

@auth_bp.route("/google/login")
@login_required_with_redirect
def google_login():
    if not is_google_configured():
        flash("La autenticación con Google no está configurada.", "warning")
        return redirect(url_for("auth.login"))
    
    google_provider_cfg = get_google_provider_cfg()
    if not google_provider_cfg:
        flash("Error de configuración con Google.", "danger")
        return redirect(url_for("auth.login"))
    
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    google_config = get_google_config()
    
    request_uri = {
        "client_id": google_config['client_id'],
        "redirect_uri": url_for("auth.google_callback", _external=True),
        "scope": "openid email profile",
        "response_type": "code",
        "prompt": "select_account"
    }
    
    redirect_url = f"{authorization_endpoint}?{urlencode(request_uri)}"
    return redirect(redirect_url)

@auth_bp.route("/google/callback")
def google_callback():
    try:
        if not is_google_configured():
            flash("La autenticación con Google no está configurada.", "warning")
            return redirect(url_for("auth.login"))
        
        code = request.args.get("code")
        if not code:
            error = request.args.get("error")
            error_description = request.args.get("error_description", "Error desconocido")
            logger.error(f"Google OAuth error: {error} - {error_description}")
            flash("Error en la autenticación con Google. Por favor intenta nuevamente.", "danger")
            return redirect(url_for("auth.login"))
        
        google_provider_cfg = get_google_provider_cfg()
        if not google_provider_cfg:
            flash("Error de configuración con Google.", "danger")
            return redirect(url_for("auth.login"))
        
        token_endpoint = google_provider_cfg["token_endpoint"]
        userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
        google_config = get_google_config()
        
        token_url = token_endpoint
        token_data = {
            "client_id": google_config['client_id'],
            "client_secret": google_config['client_secret'],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url_for("auth.google_callback", _external=True)
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        tokens = token_response.json()
        
        userinfo_response = requests.get(
            userinfo_endpoint,
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()
        
        google_id = userinfo["sub"]
        email = userinfo["email"]
        name = userinfo.get("name", "")
        picture = userinfo.get("picture", "")
        
        user = User.get_by_email(email) or User.get_by_google_id(google_id)
        
        if not user:
            username = email.split('@')[0]
            base_username = username
            counter = 1
            
            while User.get_by_username(username):
                username = f"{base_username}{counter}"
                counter += 1
            
            try:
                user = User.create_google_user(username, email, google_id, name, picture)
            except AttributeError:
                import secrets
                temp_password = secrets.token_urlsafe(16)
                user = User.create(username, email, temp_password)
                user.google_id = google_id
                if hasattr(user, 'name'):
                    user.name = name
                if hasattr(user, 'picture'):
                    user.picture = picture
                user.save()
            
            logger.info(f"Nuevo usuario registrado con Google: {username} ({email})")
            flash("¡Cuenta creada exitosamente con Google! Bienvenido a Gestos AI.", "success")
        else:
            if not user.google_id:
                user.google_id = google_id
            if name and hasattr(user, 'name') and not user.name:
                user.name = name
            if picture and hasattr(user, 'picture') and not user.picture:
                user.picture = picture
            user.save()
            
            logger.info(f"Login exitoso con Google: {user.username}")
            flash("¡Bienvenido de nuevo! Sesión iniciada con Google.", "success")
        
        login_user(user, remember=True)
        
        return redirect(url_for("gestures_bp.dashboard"))
        
    except requests.RequestException as e:
        logger.error(f"Error en callback de Google: {str(e)}")
        flash("Error en la autenticación con Google. Por favor intenta nuevamente.", "danger")
        return redirect(url_for("auth.login"))
    except Exception as e:
        logger.error(f"Error inesperado en callback de Google: {str(e)}")
        flash("Error interno del sistema. Por favor intenta más tarde.", "danger")
        return redirect(url_for("auth.login"))

@auth_bp.route("/logout")
@login_required
def logout():
    try:
        username = current_user.username
        logout_user()
        logger.info(f"Logout exitoso para usuario: {username}")
        flash("Sesión cerrada correctamente. ¡Hasta pronto!", "info")
    except Exception as e:
        logger.error(f"Error en logout: {str(e)}")
        flash("Error al cerrar sesión.", "warning")
    
    return redirect(url_for("home"))

@auth_bp.route("/forgot-password")
@login_required_with_redirect
def forgot_password():
    return render_template("auth/forgot_password.html")

@auth_bp.route("/reset-password", methods=["GET", "POST"])
@login_required_with_redirect
def reset_password():
    try:
        if request.method == "POST":
            email = request.form.get("email", "").strip()
            
            if not email:
                flash("Por favor ingresa tu email.", "warning")
                return render_template("auth/forgot_password.html", email=email)
            
            try:
                email = validate_email(email)
            except AuthError as e:
                flash(str(e), "warning")
                return render_template("auth/forgot_password.html", email=email)
            
            user = User.get_by_email(email)
            if user:
                token = generate_reset_token(user.id)
                
                email_sent = send_password_reset_email(user, token)
                
                if email_sent:
                    flash("Te hemos enviado un email con instrucciones para restablecer tu contraseña.", "success")
                    logger.info(f"Proceso de recuperación iniciado para: {email}")
                else:
                    flash("Error al enviar el email. Por favor intenta más tarde o contacta al soporte.", "danger")
                    logger.error(f"Fallo envío email a: {email}")
                
                cleanup_expired_tokens()
            else:
                flash("Si el email existe, recibirás instrucciones para restablecer tu contraseña.", "info")
            
            return redirect(url_for("auth.login"))
        
        return render_template("auth/forgot_password.html")
        
    except Exception as e:
        logger.error(f"Error en reset password request: {str(e)}")
        flash("Error al procesar la solicitud. Por favor intenta más tarde.", "danger")
        return render_template("auth/forgot_password.html")

@auth_bp.route("/reset-password/<token>", methods=["GET", "POST"])
@login_required_with_redirect
def reset_password_confirm(token):
    try:
        logger.info(f"Verificando token: {token}")
        token_data = verify_reset_token(token)
        
        if not token_data:
            logger.error(f"Token inválido o expirado: {token}")
            return render_template("auth/reset_password.html", token_valid=False)
        
        user_id = token_data['user_id']
        
        user = None
        if hasattr(User, 'get_by_id'):
            user = User.get_by_id(user_id)
        elif hasattr(User, 'get'):
            user = User.get(user_id)
        else:
            logger.error("No se encontró método para obtener usuario por ID")
            return render_template("auth/reset_password.html", token_valid=False)
        
        if not user:
            logger.error(f"Usuario no encontrado para ID: {user_id}")
            return render_template("auth/reset_password.html", token_valid=False)
        
        logger.info(f"Token válido para usuario: {user.username}")
        
        if request.method == "POST":
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            
            errors = {}
            
            try:
                password = validate_password(password)
            except AuthError as e:
                errors["password"] = str(e)
            
            if password != confirm_password:
                errors["confirm_password"] = "Las contraseñas no coinciden"
            
            if errors:
                return render_template("auth/reset_password.html", token_valid=True, token=token, error=errors)
            
            try:
                if hasattr(user, 'set_password'):
                    user.set_password(password)
                else:
                    user.password = generate_password_hash(password)
                
                user.save()
                mark_token_used(token)
                
                logger.info(f"Contraseña actualizada exitosamente para: {user.username}")
                flash("¡Contraseña actualizada exitosamente! Ya puedes iniciar sesión con tu nueva contraseña.", "success")
                return redirect(url_for("auth.login"))
                
            except Exception as e:
                logger.error(f"Error actualizando contraseña para {user.username}: {str(e)}")
                flash("Error al actualizar la contraseña. Por favor intenta nuevamente.", "danger")
                return render_template("auth/reset_password.html", token_valid=True, token=token)
        
        return render_template("auth/reset_password.html", token_valid=True, token=token)
        
    except Exception as e:
        logger.error(f"Error crítico en reset_password_confirm: {str(e)}", exc_info=True)
        flash("Error interno del sistema. Por favor solicita un nuevo enlace de recuperación.", "danger")
        return redirect(url_for("auth.forgot_password"))

@auth_bp.route("/profile")
@login_required
def profile():
    return render_template("auth/profile.html", user=current_user)

@auth_bp.errorhandler(AuthError)
def handle_auth_error(e):
    return handle_auth_error(e)

@auth_bp.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@auth_bp.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500