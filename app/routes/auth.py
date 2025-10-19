from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
import re
import logging
from datetime import datetime, timedelta
from functools import wraps
import requests
from urllib.parse import urlencode

from app.models import User

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)

# Constantes de configuración
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_TIME = timedelta(minutes=15)
PASSWORD_MIN_LENGTH = 8

# Configuración de Google OAuth (se cargará dinámicamente)
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

class AuthError(Exception):
    """Excepción personalizada para errores de autenticación"""
    pass

def get_google_provider_cfg():
    """Obtiene la configuración del proveedor Google OAuth"""
    try:
        response = requests.get(GOOGLE_DISCOVERY_URL)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error obteniendo configuración Google OAuth: {e}")
        return None

def get_google_config():
    """Obtiene la configuración de Google de forma segura"""
    try:
        return {
            'client_id': current_app.config.get('GOOGLE_CLIENT_ID', ''),
            'client_secret': current_app.config.get('GOOGLE_CLIENT_SECRET', '')
        }
    except RuntimeError:
        # Fuera del contexto de la aplicación
        return {'client_id': '', 'client_secret': ''}

def is_google_configured():
    """Verifica si Google OAuth está configurado"""
    config = get_google_config()
    return bool(config['client_id'] and config['client_secret'])

def validate_username(username):
    """Valida que el nombre de usuario cumpla con los requisitos"""
    if not username or len(username.strip()) < 3:
        raise AuthError("El nombre de usuario debe tener al menos 3 caracteres")
    
    if len(username) > 50:
        raise AuthError("El nombre de usuario no puede tener más de 50 caracteres")
    
    # Solo permite letras, números, guiones y guiones bajos
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise AuthError("El nombre de usuario solo puede contener letras, números, guiones y guiones bajos")
    
    return username.strip()

def validate_email(email):
    """Valida que el email tenga formato correcto"""
    if not email:
        raise AuthError("El email es requerido")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise AuthError("Por favor ingresa un email válido")
    
    return email.strip().lower()

def validate_password(password):
    """Valida que la contraseña cumpla con los requisitos de seguridad"""
    if not password or len(password) < PASSWORD_MIN_LENGTH:
        raise AuthError(f"La contraseña debe tener al menos {PASSWORD_MIN_LENGTH} caracteres")
    
    # Verificar fortaleza de contraseña
    if len(password) < 12:
        # Contraseñas cortas requieren más complejidad
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        if not (has_upper and has_lower and has_digit):
            raise AuthError("La contraseña debe contener mayúsculas, minúsculas y números")
    
    return password

def handle_auth_error(error):
    """Maneja errores de autenticación de forma consistente"""
    logger.warning(f"Error de autenticación: {error}")
    flash(str(error), "danger")
    return None

def check_brute_force_prevention(user):
    """Verifica y maneja la prevención de ataques de fuerza bruta"""
    if not user:
        return True
    
    if hasattr(user, 'login_attempts') and hasattr(user, 'locked_until'):
        if user.locked_until and user.locked_until > datetime.utcnow():
            remaining_time = user.locked_until - datetime.utcnow()
            minutes = int(remaining_time.total_seconds() / 60)
            raise AuthError(f"Cuenta temporalmente bloqueada. Intenta nuevamente en {minutes} minutos")
        
        if user.login_attempts >= MAX_LOGIN_ATTEMPTS:
            # Bloquear la cuenta
            user.locked_until = datetime.utcnow() + LOCKOUT_TIME
            user.save()
            raise AuthError("Demasiados intentos fallidos. Cuenta bloqueada temporalmente")
    
    return True

def update_login_attempts(user, success):
    """Actualiza los intentos de login"""
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
    """Decorator personalizado para redirigir usuarios autenticados"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated:
            flash("Ya tienes una sesión activa.", "info")
            return redirect(url_for("gestures_bp.dashboard"))

        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route("/login", methods=["GET", "POST"])
@login_required_with_redirect
def login():
    """Maneja el inicio de sesión de usuarios"""
    try:
        if request.method == "POST":
            # Validaciones básicas
            if not request.form.get("username") or not request.form.get("password"):
                flash("Por favor completa todos los campos requeridos.", "warning")
                return render_template("auth/login.html", 
                                    google_enabled=is_google_configured())
            
            username = request.form["username"].strip()
            password = request.form["password"]
            remember = bool(request.form.get("remember"))
            
            # Buscar usuario
            user = User.get_by_username(username)
            
            # Prevención de fuerza bruta
            try:
                check_brute_force_prevention(user)
            except AuthError as e:
                return render_template("auth/login.html", 
                                    username=username,
                                    error={"general": str(e)},
                                    google_enabled=is_google_configured())
            
            # Verificar credenciales
            if user and user.check_password(password):
                # Login exitoso
                login_user(user, remember=remember)
                update_login_attempts(user, True)
                
                logger.info(f"Login exitoso para usuario: {username}")
                flash("¡Bienvenido de nuevo! Sesión iniciada correctamente.", "success")
                
                # Redirigir a la página solicitada o al dashboard
                next_page = request.args.get('next')
                if next_page and next_page.startswith('/'):
                    return redirect(next_page)
                return redirect(url_for("gestures.dashboard"))
            else:
                # Login fallido
                update_login_attempts(user, False)
                remaining_attempts = MAX_LOGIN_ATTEMPTS - getattr(user, 'login_attempts', 1) if user else MAX_LOGIN_ATTEMPTS - 1
                
                error_msg = "Usuario o contraseña incorrectos."
                if remaining_attempts <= 3:
                    error_msg += f" Te quedan {remaining_attempts} intentos."
                
                logger.warning(f"Intento de login fallido para usuario: {username}")
                return render_template("auth/login.html", 
                                    username=username,
                                    error={"general": error_msg},
                                    google_enabled=is_google_configured())
        
        # GET request - mostrar formulario
        return render_template("auth/login.html", 
                             google_enabled=is_google_configured())
        
    except Exception as e:
        logger.error(f"Error en login: {str(e)}", exc_info=True)
        flash("Error interno del sistema. Por favor intenta más tarde.", "danger")
        return render_template("auth/login.html", 
                             google_enabled=is_google_configured())

@auth_bp.route("/register", methods=["GET", "POST"])
@login_required_with_redirect
def register():
    """Maneja el registro de nuevos usuarios"""
    try:
        if request.method == "POST":
            # Recoger datos del formulario
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            terms = request.form.get("terms")
            
            errors = {}
            
            # Validaciones
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
            
            # Validaciones adicionales
            if password != confirm_password:
                errors["confirm_password"] = "Las contraseñas no coinciden"
            
            if not terms:
                errors["terms"] = "Debes aceptar los términos y condiciones"
            
            # Verificar si el usuario ya existe
            if not errors.get("username") and User.get_by_username(username):
                errors["username"] = "Este nombre de usuario ya está en uso"
            
            if not errors.get("email") and User.get_by_email(email):
                errors["email"] = "Este email ya está registrado"
            
            # Si hay errores, mostrar formulario con errores
            if errors:
                return render_template("auth/register.html",
                                    username=username,
                                    email=email,
                                    error=errors,
                                    google_enabled=is_google_configured())
            
            # Crear usuario
            try:
                user = User.create(username, email, password)
                logger.info(f"Nuevo usuario registrado: {username} ({email})")
                
                # Iniciar sesión automáticamente
                login_user(user)
                flash("¡Cuenta creada exitosamente! Bienvenido a Gestos AI.", "success")
                return redirect(url_for("gestures.dashboard"))
                
            except Exception as e:
                logger.error(f"Error creando usuario: {str(e)}", exc_info=True)
                flash("Error al crear la cuenta. Por favor intenta nuevamente.", "danger")
                return render_template("auth/register.html",
                                    username=username,
                                    email=email,
                                    google_enabled=is_google_configured())
        
        # GET request - mostrar formulario
        return render_template("auth/register.html", 
                             google_enabled=is_google_configured())
        
    except Exception as e:
        logger.error(f"Error en registro: {str(e)}", exc_info=True)
        flash("Error interno del sistema. Por favor intenta más tarde.", "danger")
        return render_template("auth/register.html", 
                             google_enabled=is_google_configured())

@auth_bp.route("/google/login")
@login_required_with_redirect
def google_login():
    """Inicia el proceso de autenticación con Google"""
    if not is_google_configured():
        flash("La autenticación con Google no está configurada.", "warning")
        return redirect(url_for("auth.login"))
    
    # Obtener la configuración de Google
    google_provider_cfg = get_google_provider_cfg()
    if not google_provider_cfg:
        flash("Error de configuración con Google.", "danger")
        return redirect(url_for("auth.login"))
    
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    google_config = get_google_config()
    
    # Preparar parámetros para la solicitud
    request_uri = {
        "client_id": google_config['client_id'],
        "redirect_uri": url_for("auth.google_callback", _external=True),
        "scope": "openid email profile",
        "response_type": "code",
        "prompt": "select_account"
    }
    
    # Redirigir a Google
    redirect_url = f"{authorization_endpoint}?{urlencode(request_uri)}"
    return redirect(redirect_url)

@auth_bp.route("/google/callback")
def google_callback():
    """Maneja la respuesta de Google OAuth"""
    try:
        if not is_google_configured():
            flash("La autenticación con Google no está configurada.", "warning")
            return redirect(url_for("auth.login"))
        
        # Obtener el código de autorización
        code = request.args.get("code")
        if not code:
            error = request.args.get("error")
            error_description = request.args.get("error_description", "Error desconocido")
            logger.error(f"Google OAuth error: {error} - {error_description}")
            flash("Error en la autenticación con Google. Por favor intenta nuevamente.", "danger")
            return redirect(url_for("auth.login"))
        
        # Obtener la configuración de Google
        google_provider_cfg = get_google_provider_cfg()
        if not google_provider_cfg:
            flash("Error de configuración con Google.", "danger")
            return redirect(url_for("auth.login"))
        
        token_endpoint = google_provider_cfg["token_endpoint"]
        userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
        google_config = get_google_config()
        
        # Intercambiar el código por un token de acceso
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
        
        # Obtener información del usuario
        userinfo_response = requests.get(
            userinfo_endpoint,
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()
        
        # Extraer información del usuario
        google_id = userinfo["sub"]
        email = userinfo["email"]
        name = userinfo.get("name", "")
        picture = userinfo.get("picture", "")
        
        # Buscar usuario por email o google_id
        user = User.get_by_email(email) or User.get_by_google_id(google_id)
        
        if not user:
            # Crear nuevo usuario con Google
            username = email.split('@')[0]
            base_username = username
            counter = 1
            
            # Asegurar que el username sea único
            while User.get_by_username(username):
                username = f"{base_username}{counter}"
                counter += 1
            
            # Crear usuario para Google - necesitarás ajustar tu modelo User
            try:
                # Si tu modelo User tiene un método para crear usuarios Google
                user = User.create_google_user(username, email, google_id, name, picture)
            except AttributeError:
                # Fallback: crear usuario normal con contraseña aleatoria
                import secrets
                temp_password = secrets.token_urlsafe(16)
                user = User.create(username, email, temp_password)
                user.google_id = google_id
                user.name = name
                user.picture = picture
                user.save()
            
            logger.info(f"Nuevo usuario registrado con Google: {username} ({email})")
            flash("¡Cuenta creada exitosamente con Google! Bienvenido a Gestos AI.", "success")
        else:
            # Actualizar información de Google si es necesario
            if not user.google_id:
                user.google_id = google_id
            if name and not user.name:
                user.name = name
            if picture and not user.picture:
                user.picture = picture
            user.save()
            
            logger.info(f"Login exitoso con Google: {user.username}")
            flash("¡Bienvenido de nuevo! Sesión iniciada con Google.", "success")
        
        # Iniciar sesión
        login_user(user, remember=True)
        
        return redirect(url_for("gestures.dashboard"))
        
    except requests.RequestException as e:
        logger.error(f"Error en callback de Google: {str(e)}")
        flash("Error en la autenticación con Google. Por favor intenta nuevamente.", "danger")
        return redirect(url_for("auth.login"))
    except Exception as e:
        logger.error(f"Error inesperado en callback de Google: {str(e)}", exc_info=True)
        flash("Error interno del sistema. Por favor intenta más tarde.", "danger")
        return redirect(url_for("auth.login"))

@auth_bp.route("/logout")
@login_required
def logout():
    """Cierra la sesión del usuario"""
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
    """Muestra el formulario para recuperar contraseña"""
    return render_template("auth/forgot_password.html")

@auth_bp.route("/reset-password", methods=["POST"])
@login_required_with_redirect
def reset_password():
    """Maneja el restablecimiento de contraseña"""
    try:
        email = request.form.get("email", "").strip()
        
        if not email:
            flash("Por favor ingresa tu email.", "warning")
            return redirect(url_for("auth.forgot_password"))
        
        user = User.get_by_email(email)
        if user:
            # En un sistema real, aquí enviarías un email con un token
            logger.info(f"Solicitud de reset de contraseña para: {email}")
            flash("Si el email existe, recibirás instrucciones para restablecer tu contraseña.", "info")
        else:
            # Por seguridad, no revelamos si el email existe o no
            flash("Si el email existe, recibirás instrucciones para restablecer tu contraseña.", "info")
        
        return redirect(url_for("auth.login"))
        
    except Exception as e:
        logger.error(f"Error en reset password: {str(e)}")
        flash("Error al procesar la solicitud. Por favor intenta más tarde.", "danger")
        return redirect(url_for("auth.forgot_password"))

@auth_bp.route("/profile")
@login_required
def profile():
    """Muestra el perfil del usuario"""
    return render_template("auth/profile.html", user=current_user)

# Handlers de error específicos para auth
@auth_bp.errorhandler(AuthError)
def handle_auth_error(e):
    return handle_auth_error(e)

@auth_bp.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@auth_bp.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500