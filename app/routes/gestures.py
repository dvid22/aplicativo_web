from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from bson.objectid import ObjectId
import base64
import cv2
import numpy as np
import mediapipe as mp

gestures_bp = Blueprint("gestures", __name__)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def decode_base64_image(base64_data):
    """Convierte imagen base64 a OpenCV"""
    header, encoded = base64_data.split(",", 1)
    image_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

@gestures_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)

@gestures_bp.route("/api/register_gesture", methods=["POST"])
@login_required
def register_gesture():
    mongo = current_app.extensions['pymongo']
    data = request.json
    name = data.get("name")
    img_data = data.get("image")

    if not name or not img_data:
        return jsonify({"error": "Faltan datos"}), 400

    image_rgb = decode_base64_image(img_data)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({"error": "No se detectó ninguna mano"}), 400

    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    mongo.db.users.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$push": {"gestures": {"name": name, "landmarks": landmarks}}}
    )

    return jsonify({"success": True})


@gestures_bp.route("/api/recognize_gesture", methods=["POST"])
@login_required
def recognize_gesture():
    mongo = current_app.extensions['pymongo']
    data = request.json
    img_data = data.get("image")

    if not img_data:
        return jsonify({"error": "Falta la imagen"}), 400

    image_rgb = decode_base64_image(img_data)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({"result": "No se detectó gesto"})

    input_landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        input_landmarks.extend([lm.x, lm.y, lm.z])

    user_data = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
    gestures = user_data.get("gestures", [])

    best_match = None
    min_distance = float("inf")

    for gesture in gestures:
        stored = gesture.get("landmarks", [])
        if len(stored) == len(input_landmarks):
            dist = np.linalg.norm(np.array(stored) - np.array(input_landmarks))
            if dist < min_distance:
                min_distance = dist
                best_match = gesture.get("name")

    return jsonify({"result": best_match or "Gesto desconocido"})

