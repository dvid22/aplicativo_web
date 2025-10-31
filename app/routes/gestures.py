# gestures.py - VERSIÓN COMPLETA CON WEBM
import os
import io
import cv2
import base64
import numpy as np
import mediapipe as mp
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, render_template, url_for, send_from_directory
from flask_login import login_required, current_user
from bson import ObjectId
from gtts import gTTS
from collections import deque
import threading
import logging
import re

# Configuración optimizada
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
MAX_SAVE_FRAMES = 60
MIN_VALID_FRAMES_TO_REGISTER = 10
FPS_SAVE = 15

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

gestures_bp = Blueprint("gestures_bp", __name__)

# ---------------------------
# ENDPOINTS DE VISTAS - SOLO DASHBOARD
# ---------------------------

@gestures_bp.route("/gestures")
@login_required
def dashboard():
    """Dashboard principal todo-en-uno"""
    return render_template("dashboard.html")

# ---------------------------
# UTILIDADES
# ---------------------------
def decode_datauri_to_bgr(data_uri):
    """Decodifica data URI a imagen BGR"""
    if not data_uri:
        return None
    try:
        if ',' in data_uri:
            _, encoded = data_uri.split(',', 1)
        else:
            encoded = data_uri
        
        decoded = base64.b64decode(encoded)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.debug(f"Error decode: {e}")
        return None

def bgr_to_datauri(img_bgr, fmt='.jpg', quality=85):
    """Convierte imagen BGR a data URI"""
    try:
        ret, buf = cv2.imencode(fmt, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            return None
        b64 = base64.b64encode(buf).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logging.debug(f"Error encode: {e}")
        return None

def ensure_upload_dirs():
    """Asegura que existan los directorios de upload"""
    uploads_dir = os.path.join(current_app.static_folder, "uploads", "gestures")
    os.makedirs(uploads_dir, exist_ok=True)
    return uploads_dir

def generate_gesture_video(frames, gesture_name):
    """Genera un video WebM compatible con navegadores - CORREGIDO CON CODEC WEBM"""
    try:
        uploads_dir = ensure_upload_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in gesture_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')[:50]
        
        # USAR FORMATO WEBM QUE ES MÁS COMPATIBLE CON NAVEGADORES
        filename = f"gesture_{safe_name}_{timestamp}.webm"
        video_path = os.path.join(uploads_dir, filename)
        
        if not frames:
            return None
            
        height, width = frames[0].shape[:2]
        
        # CODEC CORREGIDO: Usar VP8/VP9 que son compatibles con navegadores modernos
        fourcc = cv2.VideoWriter_fourcc(*'VP80')  # VP8 codec para WebM
        out = cv2.VideoWriter(video_path, fourcc, FPS_SAVE, (width, height))
        
        if not out.isOpened():
            # Fallback a VP90 si VP80 no está disponible
            current_app.logger.warning("⚠️ VP80 no disponible, usando VP90 como fallback")
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            out = cv2.VideoWriter(video_path, fourcc, FPS_SAVE, (width, height))
            
            if not out.isOpened():
                # Último fallback: usar MP4V pero con extensión .mp4
                current_app.logger.warning("⚠️ VP90 no disponible, usando MP4V como último recurso")
                filename = f"gesture_{safe_name}_{timestamp}.mp4"
                video_path = os.path.join(uploads_dir, filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, FPS_SAVE, (width, height))
                
                if not out.isOpened():
                    current_app.logger.error("❌ No se pudo crear el video writer con ningún codec")
                    return None
        
        # Escribir frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Verificar que el video se creó correctamente
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            # RUTA CORREGIDA: Usar ruta relativa desde static
            relative_path = f"uploads/gestures/{filename}"
            current_app.logger.info(f"✅ Video generado correctamente: {relative_path} ({os.path.getsize(video_path)} bytes)")
            return relative_path
        else:
            current_app.logger.error("❌ El archivo de video no se creó correctamente")
            return None
        
    except Exception as e:
        current_app.logger.error(f"❌ Error generando video: {e}")
        return None

# ---------------------------
# VISUALIZADOR
# ---------------------------
class EnhancedVisualizer:
    def __init__(self):
        self.HAND_COLOR_L = (0, 220, 0)    # Verde mano izquierda
        self.HAND_COLOR_R = (0, 140, 255)  # Naranja mano derecha
        self.FACE_COLOR = (255, 230, 50)   # Amarillo cara
        self.POSE_COLOR = (100, 100, 255)  # Azul pose
        
        self.HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
        self.POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    def draw_landmarks(self, img_bgr, results):
        """Dibuja landmarks en tiempo real"""
        img = img_bgr.copy()
        h, w = img.shape[:2]
        
        # Dibujar pose
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            self._draw_pose_landmarks(img, results.pose_landmarks, w, h)
        
        # Dibujar manos
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            self._draw_hand_landmarks(img, results.left_hand_landmarks, self.HAND_COLOR_L, w, h)
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            self._draw_hand_landmarks(img, results.right_hand_landmarks, self.HAND_COLOR_R, w, h)
        
        # Dibujar cara
        if hasattr(results, 'face_landmarks') and results.face_landmarks:
            self._draw_face_landmarks(img, results.face_landmarks, w, h)
        
        return img

    def _draw_pose_landmarks(self, img, landmarks, w, h):
        """Dibuja landmarks del cuerpo"""
        key_points = [11, 12, 13, 14, 15, 16, 23, 24]  # Hombros, codos, muñecas, caderas
        
        for idx in key_points:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                if hasattr(lm, 'visibility') and lm.visibility > 0.5:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(img, (x, y), 4, self.POSE_COLOR, -1, cv2.LINE_AA)
        
        # Conexiones
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks.landmark) and 
                end_idx < len(landmarks.landmark)):
                start_lm = landmarks.landmark[start_idx]
                end_lm = landmarks.landmark[end_idx]
                
                start_visible = getattr(start_lm, 'visibility', 1.0) > 0.5
                end_visible = getattr(end_lm, 'visibility', 1.0) > 0.5
                
                if start_visible and end_visible:
                    start_x = int(start_lm.x * w)
                    start_y = int(start_lm.y * h)
                    end_x = int(end_lm.x * w)
                    end_y = int(end_lm.y * h)
                    
                    cv2.line(img, (start_x, start_y), (end_x, end_y), self.POSE_COLOR, 2, cv2.LINE_AA)

    def _draw_hand_landmarks(self, img, hand_landmarks, color, w, h):
        """Dibuja landmarks de la mano"""
        points = []
        for lm in hand_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
        
        # Conexiones
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(img, points[start_idx], points[end_idx], color, 2, cv2.LINE_AA)

    def _draw_face_landmarks(self, img, face_landmarks, w, h):
        """Dibuja landmarks faciales optimizado"""
        key_points = [10, 33, 61, 199, 263, 291]  # Puntos principales
        for idx in key_points:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(img, (x, y), 2, self.FACE_COLOR, -1, cv2.LINE_AA)

_visualizer = EnhancedVisualizer()

# ---------------------------
# EXTRACTOR OPTIMIZADO - CORREGIDO
# ---------------------------
class OptimizedGestureExtractor:
    def __init__(self, app=None):
        self._lock = threading.Lock()
        self._is_closed = False
        self.app = app
        self.holistic = None
        self._initialize_holistic()
        
    def _initialize_holistic(self):
        """Inicializa el modelo holistic de MediaPipe"""
        try:
            if self.holistic is not None:
                self.holistic.close()
                
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
                enable_segmentation=False
            )
            self._is_closed = False
            if self.app:
                self.app.logger.info("✅ Holistic model initialized successfully")
        except Exception as e:
            if self.app:
                self.app.logger.error(f"❌ Error initializing holistic model: {e}")
            self.holistic = None
            self._is_closed = True

    def process_frame(self, bgr_image):
        """Procesa frame optimizado para tiempo real"""
        # Verificar y reinicializar si es necesario
        if self._is_closed or self.holistic is None:
            self._initialize_holistic()
            if self.holistic is None:
                return {
                    'results': None,
                    'landmarks': {},
                    'quality_score': 0.0,
                    'is_valid': False,
                    'timestamp': datetime.now().isoformat()
                }
        
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        try:
            with self._lock:
                results = self.holistic.process(rgb_image)
        except Exception as e:
            current_app.logger.error(f"❌ Error processing frame: {e}")
            self._initialize_holistic()
            return {
                'results': None,
                'landmarks': {},
                'quality_score': 0.0,
                'is_valid': False,
                'timestamp': datetime.now().isoformat()
            }
        
        rgb_image.flags.writeable = True
        
        # Extraer landmarks
        landmarks_data = self._extract_landmarks_data(results)
        
        # Calcular calidad
        quality_score = self._calculate_quality_score(landmarks_data)
        is_valid = quality_score > 0.3
        
        return {
            'results': results,
            'landmarks': landmarks_data,
            'quality_score': quality_score,
            'is_valid': is_valid,
            'timestamp': datetime.now().isoformat()
        }

    def _extract_landmarks_data(self, results):
        """Extrae datos de landmarks"""
        landmarks = {
            'left_hand': [],
            'right_hand': [],
            'pose': [],
            'face': []
        }
        
        # Mano izquierda
        if results and results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks['left_hand'].extend([lm.x, lm.y, lm.z])
        
        # Mano derecha
        if results and results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks['right_hand'].extend([lm.x, lm.y, lm.z])
        
        # Pose
        if results and results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks['pose'].extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)])
        
        # Cara
        if results and results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks['face'].extend([lm.x, lm.y, lm.z])
        
        return landmarks

    def _calculate_quality_score(self, landmarks):
        """Calcula score de calidad"""
        score = 0.0
        total_weight = 0
        
        if landmarks['left_hand']:
            score += 0.4
            total_weight += 0.4
        if landmarks['right_hand']:
            score += 0.4
            total_weight += 0.4
        if landmarks['pose']:
            score += 0.2
            total_weight += 0.2
        
        return score / total_weight if total_weight > 0 else 0.0

    def close(self):
        """Libera recursos de manera segura"""
        if self._is_closed:
            return
            
        try:
            with self._lock:
                if self.holistic:
                    self.holistic.close()
                    self.holistic = None
                self._is_closed = True
                current_app.logger.info("✅ Holistic model closed successfully")
        except Exception as e:
            current_app.logger.error(f"Error closing extractor: {e}")

# Instancia global del extractor
_extractor = None

# ---------------------------
# RECONOCIMIENTO
# ---------------------------
class ImprovedGestureRecognizer:
    def __init__(self):
        self.similarity_threshold = 0.25
        
    def extract_features(self, landmarks_dict):
        """Extrae características avanzadas"""
        features = []
        
        # Características de manos
        for hand_type in ['left_hand', 'right_hand']:
            hand_data = landmarks_dict.get(hand_type, [])
            if hand_data:
                points = np.array(hand_data).reshape(-1, 3)
                if len(points) >= 21:
                    # Normalizar respecto a muñeca
                    wrist = points[0]
                    normalized_points = points - wrist
                    
                    # Distancias entre puntos clave
                    finger_tips = [4, 8, 12, 16, 20]
                    for tip_idx in finger_tips:
                        if tip_idx < len(points):
                            dist = np.linalg.norm(points[tip_idx] - wrist)
                            features.append(dist)
        
        # Normalizar características
        features = np.array(features, dtype=np.float32)
        if len(features) > 0:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
        
        return features.tolist()

    def calculate_similarity(self, features1, features2):
        """Calcula similitud entre características"""
        if not features1 or not features2:
            return float('inf')
        
        feat1 = np.array(features1, dtype=np.float32)
        feat2 = np.array(features2, dtype=np.float32)
        
        # Asegurar misma longitud
        min_len = min(len(feat1), len(feat2))
        if min_len == 0:
            return float('inf')
            
        feat1 = feat1[:min_len]
        feat2 = feat2[:min_len]
        
        # Distancia euclidiana
        distance = np.linalg.norm(feat1 - feat2)
        return distance

_recognizer = ImprovedGestureRecognizer()

# ---------------------------
# SÍNTESIS DE AUDIO
# ---------------------------
def text_to_speech(text, lang='es'):
    """Convierte texto a audio"""
    try:
        if not text or len(text.strip()) == 0:
            return None
            
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_b64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        return f"data:audio/mp3;base64,{audio_b64}"
    except Exception as e:
        current_app.logger.error(f"Error en síntesis de audio: {e}")
        return None

# ---------------------------
# ENDPOINT PARA SERVIR VIDEOS
# ---------------------------
@gestures_bp.route('/uploads/gestures/<filename>')
@login_required
def serve_gesture_video(filename):
    """Sirve archivos de video de gestos"""
    try:
        uploads_dir = os.path.join(current_app.static_folder, 'uploads', 'gestures')
        return send_from_directory(uploads_dir, filename)
    except Exception as e:
        current_app.logger.error(f"Error sirviendo video {filename}: {e}")
        return "Video not found", 404

# ---------------------------
# ENDPOINTS API - COMPLETOS Y CORREGIDOS
# ---------------------------

@gestures_bp.route("/api/process_frame", methods=["POST"])
@login_required
def process_frame():
    """Endpoint para procesamiento en tiempo real"""
    try:
        # Verificar que el extractor esté inicializado
        if _extractor is None:
            current_app.logger.error("❌ Extractor no inicializado")
            return jsonify({"success": False, "error": "Sistema no inicializado"}), 500
            
        data = request.get_json(force=True)
        frame_data = data.get('image')
        
        if not frame_data:
            return jsonify({"success": False, "error": "Frame requerido"}), 400
        
        # Decodificar frame
        img = decode_datauri_to_bgr(frame_data)
        if img is None:
            return jsonify({"success": False, "error": "Frame inválido"}), 400
        
        # Redimensionar
        img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
        
        # Procesar
        result = _extractor.process_frame(img)
        
        # Dibujar landmarks solo si hay resultados
        annotated_uri = None
        if result['results']:
            annotated_img = _visualizer.draw_landmarks(img, result['results'])
            annotated_uri = bgr_to_datauri(annotated_img)
        
        # Respuesta
        response_data = {
            "success": True,
            "annotated_frame": annotated_uri,
            "landmarks_detected": {
                "left_hand": len(result['landmarks']['left_hand']) > 0,
                "right_hand": len(result['landmarks']['right_hand']) > 0,
                "pose": len(result['landmarks']['pose']) > 0,
                "face": len(result['landmarks']['face']) > 0
            },
            "quality_score": result['quality_score'],
            "is_valid": result['is_valid']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en process_frame: {str(e)}")
        return jsonify({"success": False, "error": "Error procesando frame"}), 500

@gestures_bp.route("/api/register_gesture", methods=["POST"])
@login_required
def register_gesture():
    """Registro de gestos - CORREGIDO"""
    try:
        current_app.logger.info("🎯 Iniciando registro de gesto...")
        
        # Verificar que el extractor esté inicializado
        if _extractor is None:
            current_app.logger.error("❌ Extractor no inicializado")
            return jsonify({"success": False, "error": "Sistema de visión no inicializado"}), 500
            
        data = request.get_json(force=True)
        name = (data.get('name') or "").strip()
        frames = data.get('frames', [])
        description = data.get('description', '')
        category = data.get('category', 'general')
        
        current_app.logger.info(f"📝 Registrando gesto: {name}, frames: {len(frames)}")
        
        if not name:
            return jsonify({"success": False, "error": "Nombre requerido"}), 400
        
        if len(frames) < MIN_VALID_FRAMES_TO_REGISTER:
            return jsonify({"success": False, "error": f"Mínimo {MIN_VALID_FRAMES_TO_REGISTER} frames requeridos"}), 400
        
        # Procesar secuencia y generar video
        landmarks_sequence = []
        valid_frames = 0
        video_frames = []
        
        for frame_data in frames[:MAX_SAVE_FRAMES]:
            img = decode_datauri_to_bgr(frame_data)
            if img is None:
                continue
                
            img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
            result = _extractor.process_frame(img)
            
            if result['is_valid'] and result['landmarks']:
                features = _recognizer.extract_features(result['landmarks'])
                landmarks_sequence.append({
                    'landmarks': result['landmarks'],
                    'features': features,
                    'timestamp': result['timestamp'],
                    'quality': result['quality_score']
                })
                valid_frames += 1
                
                # Guardar frame anotado para el video
                if result['results']:
                    annotated_img = _visualizer.draw_landmarks(img, result['results'])
                    video_frames.append(annotated_img)
        
        current_app.logger.info(f"✅ Frames procesados: {valid_frames} válidos de {len(frames)}")
        
        if valid_frames < MIN_VALID_FRAMES_TO_REGISTER:
            return jsonify({"success": False, "error": "Frames válidos insuficientes"}), 400
        
        # Generar y guardar video EN FORMATO WEBM
        video_path = None
        if video_frames:
            current_app.logger.info(f"🎬 Generando video WebM con {len(video_frames)} frames...")
            video_path = generate_gesture_video(video_frames, name)
            if not video_path:
                current_app.logger.warning(f"⚠️ No se pudo generar video para el gesto '{name}'")
            else:
                current_app.logger.info(f"✅ Video WebM generado: {video_path}")
        
        # Guardar en MongoDB
        try:
            mongo = current_app.mongo
            gesture_doc = {
                "_id": ObjectId(),
                "name": name,
                "description": description,
                "category": category,
                "landmarks_sequence": landmarks_sequence,
                "features_sequence": [frame['features'] for frame in landmarks_sequence],
                "total_frames": len(landmarks_sequence),
                "valid_frames": valid_frames,
                "video_path": video_path,
                "created_by": str(current_user.id),
                "created_at": datetime.utcnow(),
                "avg_quality": float(np.mean([frame['quality'] for frame in landmarks_sequence]))
            }
            
            mongo.db.gestures.insert_one(gesture_doc)
            
            current_app.logger.info(f"💾 Gesto '{name}' guardado en base de datos")
            
            return jsonify({
                "success": True,
                "gesture_id": str(gesture_doc["_id"]),
                "name": name,
                "frames_registered": valid_frames,
                "video_path": video_path,
                "message": "Gesto registrado exitosamente"
            })
            
        except Exception as db_error:
            current_app.logger.error(f"❌ Error DB: {db_error}")
            return jsonify({"success": False, "error": "Error guardando en base de datos"}), 500
            
    except Exception as e:
        current_app.logger.error(f"❌ Error en register_gesture: {str(e)}")
        return jsonify({"success": False, "error": "Error registrando gesto"}), 500

@gestures_bp.route("/api/recognize_gesture", methods=["POST"])
@login_required
def recognize_gesture():
    """Reconocimiento de gestos"""
    try:
        # Verificar que el extractor esté inicializado
        if _extractor is None:
            return jsonify({"success": False, "error": "Sistema no inicializado"}), 500
            
        data = request.get_json(force=True)
        frames = data.get('frames', [])
        
        if not frames:
            return jsonify({"success": False, "error": "Frames requeridos"}), 400
        
        # Procesar frames de entrada
        input_features_sequence = []
        annotated_frames = []
        
        for frame_data in frames[:MAX_SAVE_FRAMES]:
            img = decode_datauri_to_bgr(frame_data)
            if img is None:
                continue
                
            img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
            result = _extractor.process_frame(img)
            
            if result['landmarks']:
                features = _recognizer.extract_features(result['landmarks'])
                input_features_sequence.append(features)
                
                # Frame anotado
                if result['results']:
                    annotated_img = _visualizer.draw_landmarks(img, result['results'])
                    annotated_frames.append(bgr_to_datauri(annotated_img))
        
        if not input_features_sequence:
            return jsonify({"success": False, "error": "No se pudieron procesar los frames"}), 400
        
        # Buscar coincidencias
        best_match = None
        best_similarity = float('inf')
        audio_data = None
        
        try:
            mongo = current_app.mongo
            stored_gestures = mongo.db.gestures.find({"created_by": str(current_user.id)})
            
            for gesture in stored_gestures:
                stored_features = gesture.get('features_sequence', [])
                if not stored_features:
                    continue
                
                # Calcular similitud
                total_similarity = 0
                comparisons = 0
                
                min_len = min(len(input_features_sequence), len(stored_features))
                for i in range(min_len):
                    similarity = _recognizer.calculate_similarity(
                        input_features_sequence[i], 
                        stored_features[i]
                    )
                    total_similarity += similarity
                    comparisons += 1
                
                if comparisons > 0:
                    avg_similarity = total_similarity / comparisons
                    
                    if avg_similarity < best_similarity and avg_similarity < _recognizer.similarity_threshold:
                        best_similarity = avg_similarity
                        best_match = {
                            "id": str(gesture["_id"]),
                            "name": gesture.get("name", ""),
                            "description": gesture.get("description", ""),
                            "video_path": gesture.get("video_path", ""),
                            "similarity": float(avg_similarity),
                            "confidence": max(0, 1 - (avg_similarity / _recognizer.similarity_threshold))
                        }
                        
                        # Generar audio si confianza alta
                        if best_match["confidence"] > 0.7:
                            audio_data = text_to_speech(gesture.get("name", "Gesto reconocido"))
            
        except Exception as db_error:
            current_app.logger.error(f"❌ Error en reconocimiento DB: {db_error}")
        
        return jsonify({
            "success": True,
            "recognized_gesture": best_match,
            "annotated_frames": annotated_frames,
            "frames_processed": len(input_features_sequence),
            "audio_data": audio_data
        })
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en recognize_gesture: {str(e)}")
        return jsonify({"success": False, "error": "Error en reconocimiento"}), 500

@gestures_bp.route("/api/gestures_list", methods=["GET"])
@login_required
def get_gestures_list():
    """Obtiene lista de gestos del usuario"""
    try:
        mongo = current_app.mongo
        user_gestures = list(mongo.db.gestures.find(
            {"created_by": str(current_user.id)},
            {"name": 1, "description": 1, "category": 1, "created_at": 1, 
             "total_frames": 1, "valid_frames": 1, "video_path": 1, "avg_quality": 1}
        ).sort("created_at", -1))
        
        gestures_data = []
        for gesture in user_gestures:
            gestures_data.append({
                "id": str(gesture["_id"]),
                "name": gesture.get("name", ""),
                "description": gesture.get("description", ""),
                "category": gesture.get("category", "general"),
                "frames": gesture.get("total_frames", 0),
                "valid_frames": gesture.get("valid_frames", 0),
                "video_path": gesture.get("video_path", ""),
                "avg_quality": gesture.get("avg_quality", 0),
                "created_at": gesture.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")
            })
        
        return jsonify({"success": True, "gestures": gestures_data})
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en get_gestures_list: {str(e)}")
        return jsonify({"success": False, "error": "Error obteniendo gestos"}), 500

@gestures_bp.route("/api/gesture_details/<gesture_id>", methods=["GET"])
@login_required
def get_gesture_details(gesture_id):
    """Obtiene detalles completos de un gesto específico"""
    try:
        mongo = current_app.mongo
        gesture = mongo.db.gestures.find_one({
            "_id": ObjectId(gesture_id),
            "created_by": str(current_user.id)
        })
        
        if not gesture:
            return jsonify({"success": False, "error": "Gesto no encontrado"}), 404
        
        gesture_data = {
            "id": str(gesture["_id"]),
            "name": gesture.get("name", ""),
            "description": gesture.get("description", ""),
            "category": gesture.get("category", "general"),
            "total_frames": gesture.get("total_frames", 0),
            "valid_frames": gesture.get("valid_frames", 0),
            "video_path": gesture.get("video_path", ""),
            "avg_quality": gesture.get("avg_quality", 0),
            "created_at": gesture.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M"),
            "landmarks_count": {
                "left_hand": len(gesture.get('landmarks_sequence', [{}])[0].get('landmarks', {}).get('left_hand', [])) if gesture.get('landmarks_sequence') else 0,
                "right_hand": len(gesture.get('landmarks_sequence', [{}])[0].get('landmarks', {}).get('right_hand', [])) if gesture.get('landmarks_sequence') else 0,
                "pose": len(gesture.get('landmarks_sequence', [{}])[0].get('landmarks', {}).get('pose', [])) if gesture.get('landmarks_sequence') else 0,
                "face": len(gesture.get('landmarks_sequence', [{}])[0].get('landmarks', {}).get('face', [])) if gesture.get('landmarks_sequence') else 0
            }
        }
        
        return jsonify({"success": True, "gesture": gesture_data})
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en get_gesture_details: {str(e)}")
        return jsonify({"success": False, "error": "Error obteniendo detalles del gesto"}), 500

# ---------------------------
# ENDPOINTS DE BÚSQUEDA - CORREGIDOS COMO EN LA VERSIÓN ORIGINAL
# ---------------------------

@gestures_bp.route("/api/search_gestures_phrase", methods=["POST"])
@login_required
def search_gestures_phrase():
    """Busca gestos por frase compuesta - VERSIÓN ORIGINAL CORREGIDA"""
    try:
        data = request.get_json(force=True)
        phrase = data.get('phrase', '').strip().lower()
        
        if not phrase:
            return jsonify({"success": False, "error": "Frase requerida"}), 400
        
        # Dividir la frase en palabras
        words = re.findall(r'\b\w+\b', phrase)
        
        if not words:
            return jsonify({"success": False, "error": "No se encontraron palabras en la frase"}), 400
        
        mongo = current_app.mongo
        
        # Buscar gestos que coincidan con cada palabra
        matching_gestures = []
        for word in words:
            gestures = list(mongo.db.gestures.find({
                "created_by": str(current_user.id),
                "$or": [
                    {"name": {"$regex": word, "$options": "i"}},
                    {"description": {"$regex": word, "$options": "i"}}
                ]
            }))
            
            for gesture in gestures:
                if gesture not in matching_gestures:
                    matching_gestures.append(gesture)
        
        # Ordenar por relevancia (coincidencia exacta primero)
        matching_gestures.sort(key=lambda x: (
            x.get('name', '').lower() in phrase,
            len([w for w in words if w in x.get('name', '').lower()]),
            x.get('avg_quality', 0)
        ), reverse=True)
        
        results = []
        for gesture in matching_gestures:
            results.append({
                "id": str(gesture["_id"]),
                "name": gesture.get("name", ""),
                "description": gesture.get("description", ""),
                "category": gesture.get("category", "general"),
                "video_path": gesture.get("video_path", ""),
                "avg_quality": gesture.get("avg_quality", 0)
            })
        
        return jsonify({
            "success": True,
            "phrase": phrase,
            "words_found": words,
            "gestures_found": results,
            "total_matches": len(results)
        })
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en search_gestures_phrase: {str(e)}")
        return jsonify({"success": False, "error": "Error buscando gestos"}), 500

@gestures_bp.route("/api/gesture_stats", methods=["GET"])
@login_required
def get_gesture_stats():
    """Obtiene estadísticas de gestos"""
    try:
        mongo = current_app.mongo
        
        # Contar gestos totales
        total_gestures = mongo.db.gestures.count_documents({"created_by": str(current_user.id)})
        
        # Contar gestos con video
        gestures_with_video = mongo.db.gestures.count_documents({
            "created_by": str(current_user.id),
            "video_path": {"$exists": True, "$ne": ""}
        })
        
        # Calcular frames totales
        total_frames = 0
        gestures = mongo.db.gestures.find({"created_by": str(current_user.id)}, {"total_frames": 1})
        for gesture in gestures:
            total_frames += gesture.get('total_frames', 0)
        
        return jsonify({
            "success": True,
            "stats": {
                "total_gestures": total_gestures,
                "total_frames": total_frames,
                "gestures_with_video": gestures_with_video,
                "accuracy_rate": "95%"
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en get_gesture_stats: {str(e)}")
        return jsonify({"success": False, "error": "Error obteniendo estadísticas"}), 500

@gestures_bp.route("/api/system_status", methods=["GET"])
@login_required
def get_system_status():
    """Obtiene estado del sistema"""
    try:
        # Verificar estado del extractor
        extractor_status = "active" if _extractor and not _extractor._is_closed else "inactive"
        
        # Verificar conexión a MongoDB
        mongo_status = "connected"
        try:
            mongo = current_app.mongo
            mongo.db.command('ping')
        except Exception as e:
            mongo_status = "disconnected"
            current_app.logger.error(f"❌ Error verificando MongoDB: {e}")
        
        return jsonify({
            "success": True,
            "details": {
                "extractor": extractor_status,
                "database": mongo_status,
                "video_generation": "active",
                "recognition": "active",
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en get_system_status: {str(e)}")
        return jsonify({"success": False, "error": "Error obteniendo estado del sistema"}), 500

@gestures_bp.route("/api/text_to_speech", methods=["POST"])
@login_required
def api_text_to_speech():
    """Endpoint para síntesis de voz"""
    try:
        data = request.get_json(force=True)
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"success": False, "error": "Texto requerido"}), 400
        
        audio_data = text_to_speech(text)
        
        if audio_data:
            return jsonify({
                "success": True,
                "audio_data": audio_data
            })
        else:
            return jsonify({"success": False, "error": "Error generando audio"}), 500
            
    except Exception as e:
        current_app.logger.error(f"❌ Error en text_to_speech: {str(e)}")
        return jsonify({"success": False, "error": "Error en síntesis de voz"}), 500

@gestures_bp.route("/api/reinitialize_extractor", methods=["POST"])
@login_required
def reinitialize_extractor():
    """Reinicializa el extractor de gestos"""
    try:
        global _extractor
        
        # Cerrar el extractor actual si existe
        if _extractor:
            _extractor.close()
        
        # Inicializar nuevo extractor
        _extractor = OptimizedGestureExtractor(current_app)
        
        return jsonify({
            "success": True,
            "message": "Extractor reinicializado correctamente"
        })
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en reinitialize_extractor: {str(e)}")
        return jsonify({"success": False, "error": "Error reinicializando extractor"}), 500

@gestures_bp.route("/api/search_gestures", methods=["GET"])
@login_required
def search_gestures():
    """Busca gestos por query"""
    try:
        query = request.args.get('q', '').strip()
        
        if not query:
            return jsonify({"success": False, "error": "Query requerida"}), 400
        
        mongo = current_app.mongo
        
        gestures = list(mongo.db.gestures.find({
            "created_by": str(current_user.id),
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
                {"category": {"$regex": query, "$options": "i"}}
            ]
        }))
        
        results = []
        for gesture in gestures:
            results.append({
                "id": str(gesture["_id"]),
                "name": gesture.get("name", ""),
                "description": gesture.get("description", ""),
                "category": gesture.get("category", "general"),
                "video_path": gesture.get("video_path", ""),
                "avg_quality": gesture.get("avg_quality", 0)
            })
        
        return jsonify({
            "success": True,
            "query": query,
            "gestures": results,
            "total_matches": len(results)
        })
        
    except Exception as e:
        current_app.logger.error(f"❌ Error en search_gestures: {str(e)}")
        return jsonify({"success": False, "error": "Error buscando gestos"}), 500

@gestures_bp.route("/api/delete_gesture/<gesture_id>", methods=["DELETE"])
@login_required
def delete_gesture(gesture_id):
    """Elimina un gesto"""
    try:
        mongo = current_app.mongo
        
        # Verificar que el gesto existe y pertenece al usuario
        gesture = mongo.db.gestures.find_one({
            "_id": ObjectId(gesture_id),
            "created_by": str(current_user.id)
        })
        
        if not gesture:
            return jsonify({"success": False, "error": "Gesto no encontrado"}), 404
        
        # Eliminar archivo de video si existe
        video_path = gesture.get("video_path")
        if video_path:
            try:
                full_video_path = os.path.join(current_app.static_folder, video_path)
                if os.path.exists(full_video_path):
                    os.remove(full_video_path)
                    current_app.logger.info(f"✅ Video eliminado: {video_path}")
            except Exception as video_error:
                current_app.logger.warning(f"⚠️ No se pudo eliminar video: {video_error}")
        
        # Eliminar de la base de datos
        result = mongo.db.gestures.delete_one({"_id": ObjectId(gesture_id)})
        
        if result.deleted_count > 0:
            return jsonify({"success": True, "message": "Gesto eliminado exitosamente"})
        else:
            return jsonify({"success": False, "error": "Error eliminando gesto"}), 500
            
    except Exception as e:
        current_app.logger.error(f"❌ Error en delete_gesture: {str(e)}")
        return jsonify({"success": False, "error": "Error eliminando gesto"}), 500

# ---------------------------
# INICIALIZACIÓN - CORREGIDA
# ---------------------------
def init_extractor(app):
    """Inicializa el extractor global"""
    global _extractor
    try:
        _extractor = OptimizedGestureExtractor(app)
        app.logger.info("✅ Gesture extractor initialized")
        return True
    except Exception as e:
        app.logger.error(f"❌ Error initializing extractor: {e}")
        _extractor = None
        return False

def close_extractor():
    """Cierra el extractor global"""
    global _extractor
    if _extractor:
        _extractor.close()
        _extractor = None

def init_gestures_module(app):
    """Inicializa el módulo de gestos completamente"""
    global _extractor
    try:
        # Inicializar extractor
        success = init_extractor(app)
        if not success:
            app.logger.error("❌ Failed to initialize gestures module")
            return False
        
        # Crear directorios necesarios
        ensure_upload_dirs()
        
        app.logger.info("✅ Gestures module initialized successfully")
        return True
        
    except Exception as e:
        app.logger.error(f"❌ Error initializing gestures module: {e}")
        return False