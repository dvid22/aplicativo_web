from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from app.extensions import mongo  # conexión Mongo

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.username = user_data.get("username")
        self.email = user_data.get("email")
        self.password_hash = user_data.get("password_hash")
        self.google_id = user_data.get("google_id")
        self.name = user_data.get("name")
        self.picture = user_data.get("picture")

    # 🔎 Obtener por ID
    @staticmethod
    def get(user_id):
        user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        return User(user_data) if user_data else None

    # 🔎 Obtener por username
    @staticmethod
    def get_by_username(username):
        user_data = mongo.db.users.find_one({"username": username})
        return User(user_data) if user_data else None

    # 🔎 Obtener por email
    @staticmethod
    def get_by_email(email):
        user_data = mongo.db.users.find_one({"email": email})
        return User(user_data) if user_data else None

    # 🔎 Obtener por Google ID
    @staticmethod
    def get_by_google_id(google_id):
        user_data = mongo.db.users.find_one({"google_id": google_id})
        return User(user_data) if user_data else None

    # ➕ Crear usuario normal
    @staticmethod
    def create(username, email, password):
        password_hash = generate_password_hash(password)
        result = mongo.db.users.insert_one({
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "gestures": [],
            "created_at": datetime.utcnow(),
            "login_attempts": 0,
            "locked_until": None
        })
        return User.get(result.inserted_id)

    # ➕ Crear usuario con Google
    @staticmethod
    def create_google_user(username, email, google_id, name=None, picture=None):
        result = mongo.db.users.insert_one({
            "username": username,
            "email": email,
            "google_id": google_id,
            "name": name,
            "picture": picture,
            "gestures": [],
            "created_at": datetime.utcnow(),
            "login_attempts": 0,
            "locked_until": None
        })
        return User.get(result.inserted_id)

    # 🔑 Verificar password
    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    # 💾 Guardar cambios
    def save(self):
        mongo.db.users.update_one(
            {"_id": ObjectId(self.id)},
            {"$set": {
                "username": self.username,
                "email": self.email,
                "password_hash": self.password_hash,
                "google_id": self.google_id,
                "name": self.name,
                "picture": self.picture
            }}
        )
