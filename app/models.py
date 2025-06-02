from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from flask import current_app

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']

    @staticmethod
    def get(user_id):
        """Obtiene un usuario por su ID desde MongoDB"""
        mongo = current_app.extensions['pymongo']
        user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        return User(user_data) if user_data else None

    @staticmethod
    def get_by_username(username):
        """Obtiene un usuario por su nombre de usuario"""
        mongo = current_app.extensions['pymongo']
        user_data = mongo.db.users.find_one({'username': username})
        return User(user_data) if user_data else None

    @staticmethod
    def create(username, email, password):
        """Crea un nuevo usuario y lo guarda en MongoDB"""
        mongo = current_app.extensions['pymongo']
        password_hash = generate_password_hash(password)
        mongo.db.users.insert_one({
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'gestures': []  # Inicializa lista de gestos vacía
        })

    def check_password(self, password):
        """Verifica la contraseña"""
        return check_password_hash(self.password_hash, password)

