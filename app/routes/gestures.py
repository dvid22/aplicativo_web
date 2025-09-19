# app/routes/gestures.py
from flask import Blueprint, request, jsonify, current_app, render_template
from flask_login import login_required, current_user
from bson.objectid import ObjectId
import base64, cv2, numpy as np, mediapipe as mp, math, difflib
import os, requests, logging
from datetime import datetime

# Importa tu cliente mongo (ajusta si tu proyecto lo expone distinto)
from app import mongo

gestures_bp = Blueprint("gestures", __name__, template_folder="../../templates")

# ----- Configuración y logging -----
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# HeyGen config desde .env
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "")   # pon aquí tu clave
HEYGEN_API_URL = os.getenv("HEYGEN_API_URL", "").rstrip("/")  # ej. https://api.heygen.com/v1/video.generate
HEYGEN_AVATAR_ID = os.getenv("HEYGEN_AVATAR_ID", "")  # opcional

# MediaPipe
mp_holistic = mp.solutions.holistic

# Singleton Holistic (para reducir overhead)
_holistic = None
def get_holistic():
    global _holistic
    if _holistic is None:
        _holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _holistic

# Dimensiones landmarks: (pose 33 + 21 left + 21 right) * 3 = 225
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
LANDMARK_DIM = 3
TOTAL_LANDMARK_SIZE = (POSE_LANDMARKS + 2 * HAND_LANDMARKS) * LANDMARK_DIM

# Umbral de similitud coseno (ajustable)
SIMILARITY_THRESHOLD = 0.70  # 0..1 (a mayor, más estricto)

# Limites
MAX_FRAMES = 60

# ----- Gestos predefinidos (ejemplo, vectores de longitud TOTAL_LANDMARK_SIZE) -----
PREDEFINED_GESTURES = {
    "hola": [0.10] * TOTAL_LANDMARK_SIZE,
    "gracias": [0.20] * TOTAL_LANDMARK_SIZE,
    "sí": [0.30] * TOTAL_LANDMARK_SIZE,
    "no": [0.40] * TOTAL_LANDMARK_SIZE,
    "comida": [0.15] * TOTAL_LANDMARK_SIZE,
    "agua": [0.25] * TOTAL_LANDMARK_SIZE,
    "ayuda": [0.35] * TOTAL_LANDMARK_SIZE,
    "baño": [0.45] * TOTAL_LANDMARK_SIZE
}

# ----- Helpers -----
def decode_datauri_to_rgb(datauri):
    """
    Convierte dataURI (data:image/...) a imagen RGB (numpy array)
    """
    try:
        if not datauri or not datauri.startswith("data:image"):
            return None
        header, encoded = datauri.split(",", 1)
        data = base64.b64decode(encoded)
        arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.exception("decode_datauri_to_rgb error: %s", e)
        return None

def extract_landmarks_from_frame(image_rgb):
    """
    Procesa un frame RGB con MediaPipe Holistic y devuelve vector plano de tamaño TOTAL_LANDMARK_SIZE.
    Si hay puntos faltantes se rellenan con 0.
    """
    try:
        holistic = get_holistic()
        results = holistic.process(image_rgb)
    except Exception as e:
        logger.exception("MediaPipe process error: %s", e)
        return None

    vec = np.zeros(TOTAL_LANDMARK_SIZE, dtype=float)
    idx = 0

    # pose 33
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            vec[idx:idx+3] = [lm.x, lm.y, lm.z]; idx += 3
    else:
        idx += POSE_LANDMARKS * LANDMARK_DIM

    # left hand 21
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            vec[idx:idx+3] = [lm.x, lm.y, lm.z]; idx += 3
    else:
        idx += HAND_LANDMARKS * LANDMARK_DIM

    # right hand 21
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            vec[idx:idx+3] = [lm.x, lm.y, lm.z]; idx += 3
    else:
        idx += HAND_LANDMARKS * LANDMARK_DIM

    return vec.tolist()

def mean_sequence(vectors):
    if not vectors:
        return None
    arr = np.array(vectors, dtype=float)
    return np.mean(arr, axis=0).tolist()

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def get_user_doc():
    """Devuelve documento de usuario (o None) manejando errores de BD"""
    try:
        return mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
    except Exception as e:
        logger.exception("DB error get_user_doc: %s", e)
        return None

def append_chat_message(user_id, sender, text):
    """Guarda mensaje en colección 'chats' (histórico)"""
    try:
        mongo.db.chats.insert_one({
            "user_id": ObjectId(user_id),
            "sender": sender,
            "text": text,
            "created_at": datetime.utcnow()
        })
    except Exception as e:
        logger.exception("Error guardando chat: %s", e)

# ----- Rutas UI -----
@gestures_bp.route("/dashboard")
@login_required
def dashboard():
    # intenta leer secuencias para el usuario para mostrarlas en la plantilla
    user_doc = get_user_doc()
    sequences = user_doc.get("sequences", []) if user_doc else []
    return render_template("dashboard.html", sequences=sequences)

# ----- API: registro de secuencia -----
@gestures_bp.route("/api/register_sequence", methods=["POST"])
@login_required
def register_sequence():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    frames = data.get("frames", [])

    if not name:
        return jsonify({"success": False, "error": "Falta 'name'"}), 400
    if not frames or not isinstance(frames, list):
        return jsonify({"success": False, "error": "Falta 'frames' (lista)"}), 400

    frames = frames[:MAX_FRAMES]
    landmarks_seq = []
    for f in frames:
        img = decode_datauri_to_rgb(f)
        if img is None:
            continue
        vec = extract_landmarks_from_frame(img)
        if vec is not None:
            landmarks_seq.append(vec)

    if not landmarks_seq:
        return jsonify({"success": False, "error": "No se extrajeron landmarks válidos"}), 400

    mean_vec = mean_sequence(landmarks_seq)
    new_seq = {
        "name": name,
        "mean_landmarks": mean_vec,
        "n_frames": len(landmarks_seq),
        "created_at": datetime.utcnow()
    }

    try:
        # push a user.sequences
        res = mongo.db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$push": {"sequences": new_seq}}
        )
        if res.modified_count == 0:
            # si el update falló (usuario no existe)
            return jsonify({"success": False, "error": "No se pudo guardar la secuencia (usuario no encontrado)"}), 500

        return jsonify({"success": True, "name": name, "n_frames": len(landmarks_seq)})
    except Exception as e:
        logger.exception("Error saving sequence: %s", e)
        return jsonify({"success": False, "error": "Error interno guardando secuencia"}), 500

# ----- API: reconocimiento continuo (single-frame o multi-frame) -----
@gestures_bp.route("/api/continuous_recognition", methods=["POST"])
@login_required
def continuous_recognition():
    """
    Recibe JSON { frames: [dataURI, ...] } (frame único o varios).
    Responde { success, found, gesture, similarity, type }
    """
    data = request.get_json(silent=True) or {}
    frames = data.get("frames", [])
    if not frames or not isinstance(frames, list):
        return jsonify({"success": False, "error": "Se requieren 'frames' (lista)"}), 400

    frames = frames[:MAX_FRAMES]
    landmarks_seq = []
    for f in frames:
        img = decode_datauri_to_rgb(f)
        if img is None:
            continue
        vec = extract_landmarks_from_frame(img)
        if vec is not None:
            landmarks_seq.append(vec)

    if not landmarks_seq:
        return jsonify({"success": True, "found": False, "message": "No landmarks detectados"}), 200

    mean_vec = mean_sequence(landmarks_seq)

    # Buscar en user sequences
    user_doc = get_user_doc()
    user_sequences = user_doc.get("sequences", []) if user_doc else []

    best = {"name": None, "sim": -1.0, "type": None}
    # user sequences
    for seq in user_sequences:
        seq_vec = seq.get("mean_landmarks")
        if seq_vec and len(seq_vec) == len(mean_vec):
            sim = cosine_similarity(mean_vec, seq_vec)
            if sim > best["sim"]:
                best.update({"name": seq.get("name"), "sim": sim, "type": "user"})

    # predefined gestures
    for name, vec in PREDEFINED_GESTURES.items():
        if len(vec) == len(mean_vec):
            sim = cosine_similarity(mean_vec, vec)
            if sim > best["sim"]:
                best.update({"name": name, "sim": sim, "type": "predefined"})

    # Resultado
    found = best["name"] is not None and best["sim"] >= SIMILARITY_THRESHOLD
    response = {
        "success": True,
        "found": found,
        "gesture": best["name"] if found else None,
        "similarity": float(best["sim"]) if best["sim"] >= 0 else 0.0,
        "type": best["type"]
    }

    # Guardar evento en histórico de chats como mensaje del sistema (opcional)
    try:
        if found:
            append_chat_message(current_user.id, "system", f"Gesto reconocido: {best['name']}")
    except Exception:
        pass

    return jsonify(response), 200

# ----- API: texto -> gesto (busca equivalente) -----
@gestures_bp.route("/api/text_to_gesture", methods=["POST"])
@login_required
def text_to_gesture():
    """
    JSON { text: "hola" } -> responde si hay gesto con ese nombre (user o predef)
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip().lower()
    if not text:
        return jsonify({"success": False, "error": "Falta texto"}), 400

    user_doc = get_user_doc()
    sequences = user_doc.get("sequences", []) if user_doc else []

    # buscar exacto en user sequences
    for seq in sequences:
        if seq.get("name", "").lower() == text:
            return jsonify({"success": True, "gesture": seq["name"], "type": "user"})

    # buscar exacto en predef
    for name in PREDEFINED_GESTURES:
        if name.lower() == text:
            return jsonify({"success": True, "gesture": name, "type": "predefined"})

    # fuzzy match (sugerencia)
    names = [s.get("name", "") for s in sequences] + list(PREDEFINED_GESTURES.keys())
    matches = difflib.get_close_matches(text, names, n=1, cutoff=0.7)
    if matches:
        match = matches[0]
        t = "user" if any(s.get("name", "") == match for s in sequences) else "predefined"
        return jsonify({"success": True, "gesture": match, "type": t, "note": "fuzzy_match"})

    return jsonify({"success": True, "gesture": None}), 200

# ----- API: generar avatar (HeyGen) -----
@gestures_bp.route("/api/avatar", methods=["POST"])
@login_required
def avatar():
    """
    JSON { text: "...", voice: "male"|"female" }
    Intenta generar un avatar con HeyGen (usando HEYGEN_API_KEY / HEYGEN_API_URL).
    Si HeyGen no está configurado o falla, devuelve un URL fallback que puedes colocar en /static.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = data.get("voice", "male")

    if not text:
        return jsonify({"success": False, "error": "Falta texto"}), 400

    # 1) intenta mapear texto a gesto (opcional)
    gesture_lookup = None
    res_g = None
    try:
        res_g = text_to_gesture_internal(text)
        if res_g:
            gesture_lookup = res_g
    except Exception:
        pass

    # 2) Llamada a HeyGen (si está configurado)
    if not HEYGEN_API_KEY or not HEYGEN_API_URL:
        # fallback: si no hay HeyGen configurado, devolvemos URL de ejemplo estático
        fallback = current_app.config.get("FALLBACK_AVATAR_URL", "/static/avatar_placeholder.mp4")
        # guardamos chat
        append_chat_message(current_user.id, "user", text)
        append_chat_message(current_user.id, "system", f"Avatar fallback para: {text}")
        return jsonify({"success": True, "gesture": gesture_lookup, "video_url": fallback, "note": "fallback"}), 200

    # Construir payload para HeyGen (ejemplo genérico — ajústalo según la especificación real)
    payload = {
        "inputs": [
            {
                "character_id": HEYGEN_AVATAR_ID or None,
                "input_text": text,
                "voice": "female" if voice == "female" else "male"
            }
        ],
        "options": {
            "resolution": "720p"
        }
    }

    headers = {
        "Authorization": f"Bearer {HEYGEN_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(HEYGEN_API_URL, json=payload, headers=headers, timeout=30)
        # validar código
        if r.status_code not in (200, 201):
            logger.error("HeyGen error %s: %s", r.status_code, r.text)
            # fallback
            fallback = current_app.config.get("FALLBACK_AVATAR_URL", "/static/avatar_placeholder.mp4")
            append_chat_message(current_user.id, "user", text)
            append_chat_message(current_user.id, "system", "Error generando avatar (HeyGen). Usando fallback.")
            return jsonify({"success": False, "error": "Error en servicio de avatar", "details": r.text, "fallback": fallback}), 502

        result = r.json()
        # La estructura de respuesta depende de HeyGen; aquí intentamos leer un campo conocido
        video_url = None
        # ejemplo genérico: {"data": {"video_url": "..."}}
        if isinstance(result, dict):
            video_url = result.get("data", {}).get("video_url") or result.get("video_url")

        if not video_url:
            # intenta buscar campos comunes en la respuesta
            # si falla, devolvemos fallback
            fallback = current_app.config.get("FALLBACK_AVATAR_URL", "/static/avatar_placeholder.mp4")
            append_chat_message(current_user.id, "user", text)
            append_chat_message(current_user.id, "system", "Avatar no retornó URL. Usando fallback.")
            return jsonify({"success": False, "error": "No video_url in response", "details": result, "fallback": fallback}), 502

        # guardar histórico avatar_request
        try:
            mongo.db.avatar_requests.insert_one({
                "user_id": ObjectId(current_user.id),
                "text": text,
                "voice": voice,
                "gesture": gesture_lookup,
                "video_url": video_url,
                "created_at": datetime.utcnow()
            })
        except Exception:
            logger.exception("Error guardando avatar request")

        # guardar en chat
        append_chat_message(current_user.id, "user", text)
        append_chat_message(current_user.id, "system", f"Avatar generado: {video_url}")

        return jsonify({"success": True, "gesture": gesture_lookup, "video_url": video_url}), 200

    except requests.exceptions.RequestException as e:
        logger.exception("HeyGen connection error: %s", e)
        fallback = current_app.config.get("FALLBACK_AVATAR_URL", "/static/avatar_placeholder.mp4")
        append_chat_message(current_user.id, "user", text)
        append_chat_message(current_user.id, "system", "Error conexión con servicio de avatar. Usando fallback.")
        return jsonify({"success": False, "error": "Error conexión HeyGen", "details": str(e), "fallback": fallback}), 503

def text_to_gesture_internal(text):
    """Helper que retorna nombre de gesto si existe (user or predef) — usado internamente."""
    if not text:
        return None
    user_doc = get_user_doc()
    sequences = user_doc.get("sequences", []) if user_doc else []
    t = text.strip().lower()
    for seq in sequences:
        if seq.get("name", "").lower() == t:
            return seq.get("name")
    for name in PREDEFINED_GESTURES.keys():
        if name.lower() == t:
            return name
    # fuzzy match
    names = [s.get("name","") for s in sequences] + list(PREDEFINED_GESTURES.keys())
    matches = difflib.get_close_matches(t, names, n=1, cutoff=0.75)
    return matches[0] if matches else None

# ----- Chat history endpoints -----
@gestures_bp.route("/api/chat_history", methods=["GET"])
@login_required
def chat_history():
    """Retorna historial de chat para el usuario (últimos 200 mensajes)"""
    try:
        cursor = mongo.db.chats.find({"user_id": ObjectId(current_user.id)}).sort("created_at", 1).limit(500)
        history = [{"sender": c.get("sender"), "text": c.get("text"), "created_at": c.get("created_at")} for c in cursor]
        return jsonify({"success": True, "history": history})
    except Exception as e:
        logger.exception("Error fetching chat history: %s", e)
        return jsonify({"success": False, "history": [], "error": "Error interno"}), 500

@gestures_bp.route("/api/chat", methods=["POST"])
@login_required
def chat_post():
    """Guardar un mensaje de chat (opcional)"""
    data = request.get_json(silent=True) or {}
    sender = data.get("sender", "user")
    text = data.get("text", "")
    if not text:
        return jsonify({"success": False, "error": "Texto vacío"}), 400
    try:
        append_chat_message(current_user.id, sender, text)
        return jsonify({"success": True})
    except Exception as e:
        logger.exception("Error saving chat: %s", e)
        return jsonify({"success": False, "error": "Error guardando chat"}), 500
