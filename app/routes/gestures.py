# gestures.py - VERSIÓN COMPLETA Y FUNCIONAL 100%
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
import time
from typing import Dict, List, Any, Optional, Tuple, Union

# ===========================
# CONFIGURACIÓN OPTIMIZADA
# ===========================
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
MAX_SAVE_FRAMES = 60
MIN_VALID_FRAMES_TO_REGISTER = 8
FPS_SAVE = 20

# Configuración de MediaPipe
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.4

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

gestures_bp = Blueprint("gestures_bp", __name__)

# ===========================
# UTILIDADES
# ===========================

def decode_datauri_to_bgr(data_uri: Optional[str]) -> Optional[np.ndarray]:
    """Decodifica data URI a imagen BGR"""
    if not data_uri:
        return None
    try:
        header, encoded = data_uri.split(',', 1) if ',' in data_uri else ('', data_uri)
        decoded = base64.b64decode(encoded)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.debug(f"Error decode: {e}")
        return None

def bgr_to_datauri(img_bgr: np.ndarray, fmt: str = '.jpg', quality: int = 85) -> Optional[str]:
    """Convierte imagen BGR a data URI"""
    try:
        ret, buf = cv2.imencode(fmt, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            return None
        b64 = base64.b64encode(buf).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.debug(f"Error encode: {e}")
        return None

def ensure_upload_dirs() -> str:
    """Asegura que existan los directorios de upload"""
    uploads_dir = os.path.join(current_app.static_folder, "uploads", "gestures")
    os.makedirs(uploads_dir, exist_ok=True)
    return uploads_dir

def generate_gesture_video(frames: List[np.ndarray], gesture_name: str) -> Optional[str]:
    """Genera un video WebM compatible"""
    try:
        if not frames:
            return None
            
        uploads_dir = ensure_upload_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[^\w\-_\. ]', '_', gesture_name)[:50].replace(' ', '_')
        
        filename = f"gesture_{safe_name}_{timestamp}.webm"
        video_path = os.path.join(uploads_dir, filename)
        
        height, width = frames[0].shape[:2]
        
        # Probar codecs en orden de compatibilidad
        codecs = ['VP80', 'VP90', 'mp4v']
        out = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(video_path, fourcc, FPS_SAVE, (width, height))
            if out.isOpened():
                break
            out = None
        
        if out is None:
            logger.error("❌ No se pudo inicializar VideoWriter")
            return None
            
        for frame in frames:
            out.write(frame)
        out.release()
        
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            relative_path = f"uploads/gestures/{filename}"
            logger.info(f"✅ Video generado: {relative_path}")
            return relative_path
        else:
            if os.path.exists(video_path):
                os.remove(video_path)
            return None
        
    except Exception as e:
        logger.error(f"❌ Error generando video: {e}")
        return None

def convert_numpy_types(obj: Any) -> Any:
    """Convierte tipos numpy a tipos nativos de Python"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ===========================
# VISUALIZADOR
# ===========================
class UltraFastVisualizer:
    def __init__(self):
        self.HAND_COLOR_L = (0, 220, 0)
        self.HAND_COLOR_R = (0, 140, 255)
        self.FACE_COLOR = (255, 230, 50)
        self.POSE_COLOR = (100, 100, 255)
        
        self.hand_style = mp_drawing_styles.get_default_hand_landmarks_style()
        self.pose_style = mp_drawing_styles.get_default_pose_landmarks_style()

    def draw_landmarks(self, img_bgr: np.ndarray, results: Any) -> np.ndarray:
        """Dibuja landmarks en tiempo real"""
        img = img_bgr.copy()
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_style
            )
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.hand_style
            )
            
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                results.right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.hand_style
            )
        
        return img

_visualizer = UltraFastVisualizer()

# ===========================
# EXTRACTOR MEJORADO
# ===========================
class HighPrecisionGestureExtractor:
    def __init__(self, app: Optional[Any] = None):
        self._lock = threading.RLock()
        self._is_closed = False
        self.app = app
        self.holistic: Optional[mp.solutions.holistic.Holistic] = None
        self.motion_history: deque[Dict[str, Any]] = deque(maxlen=10)
        self._initialize_holistic()
        
    def _initialize_holistic(self):
        """Inicializa el modelo holistic"""
        try:
            if self.holistic is not None:
                self.holistic.close()
                
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=MODEL_COMPLEXITY,
                smooth_landmarks=True,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                enable_segmentation=False
            )
            self._is_closed = False
            if self.app:
                self.app.logger.info("✅ Holistic model initialized")
        except Exception as e:
            if self.app:
                self.app.logger.error(f"❌ Error initializing holistic: {e}")
            self.holistic = None
            self._is_closed = True

    def process_frame(self, bgr_image: np.ndarray) -> Dict[str, Any]:
        """Procesa frame con máxima velocidad"""
        start_time = time.perf_counter()
        
        if self._is_closed or self.holistic is None:
            self._initialize_holistic()
            if self.holistic is None:
                return self._empty_result()

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        results = None
        try:
            with self._lock:
                results = self.holistic.process(rgb_image)
        except Exception as e:
            logger.error(f"❌ Error processing frame: {e}")
            self._initialize_holistic()
            return self._empty_result()
        
        rgb_image.flags.writeable = True
        
        landmarks_data = self._extract_landmarks_fast(results)
        motion_features = self._extract_motion_features_fast(landmarks_data)
        
        combined_features = {**landmarks_data, **motion_features}
        quality_score = self._calculate_quality_score(landmarks_data)
        is_valid = quality_score > 0.3  # Umbral más bajo para mejor detección
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'results': results,
            'landmarks': combined_features,
            'quality_score': float(quality_score),
            'is_valid': bool(is_valid),
            'timestamp': datetime.now().isoformat(),
            'has_motion_data': bool(motion_features.get('motion_vectors', {})),
            'processing_time_ms': float(processing_time)
        }

    def _extract_landmarks_fast(self, results: Any) -> Dict[str, List[float]]:
        """Extrae datos de landmarks"""
        landmarks = {
            'left_hand': [],
            'right_hand': [],
            'pose': [],
            'face': []
        }
        
        # Manos
        for hand_type, mp_landmarks in [('left_hand', results.left_hand_landmarks), 
                                        ('right_hand', results.right_hand_landmarks)]:
            if mp_landmarks:
                landmarks[hand_type] = [coord for lm in mp_landmarks.landmark 
                                      for coord in (float(lm.x), float(lm.y), float(lm.z))]
        
        # Pose - puntos clave
        pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
        if results.pose_landmarks:
            for idx in pose_indices:
                if idx < len(results.pose_landmarks.landmark):
                    lm = results.pose_landmarks.landmark[idx]
                    landmarks['pose'].extend([float(lm.x), float(lm.y), float(lm.z)])
        
        return landmarks

    def _extract_motion_features_fast(self, current_landmarks: Dict[str, List[float]]) -> Dict[str, Any]:
        """Extrae características de movimiento"""
        current_frame_data = {
            'timestamp': datetime.now(),
            'landmarks': current_landmarks
        }
        self.motion_history.append(current_frame_data)
        
        if len(self.motion_history) < 2:
            return {'motion_vectors': {}}
        
        current_frame = self.motion_history[-1]
        previous_frame = self.motion_history[-2]
        
        time_diff = (current_frame['timestamp'] - previous_frame['timestamp']).total_seconds()
        if time_diff <= 0:
            return {'motion_vectors': {}}
        
        motion_vectors = {}
        tracking_points = {
            'left_wrist': ('left_hand', 0),
            'right_wrist': ('right_hand', 0),
        }
        
        for point_name, (landmark_type, point_idx) in tracking_points.items():
            curr_data = current_frame['landmarks'].get(landmark_type, [])
            prev_data = previous_frame['landmarks'].get(landmark_type, [])
            
            idx = point_idx * 3
            if len(curr_data) > idx + 2 and len(prev_data) > idx + 2:
                curr_x, curr_y, curr_z = curr_data[idx:idx+3]
                prev_x, prev_y, prev_z = prev_data[idx:idx+3]
                
                dx, dy, dz = curr_x - prev_x, curr_y - prev_y, curr_z - prev_z
                magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
                velocity = magnitude / time_diff
                
                motion_vectors[point_name] = {
                    'dx': float(dx), 'dy': float(dy), 'dz': float(dz),
                    'magnitude': float(magnitude),
                    'velocity': float(velocity)
                }
        
        return {'motion_vectors': motion_vectors}

    def _calculate_quality_score(self, landmarks: Dict[str, List[float]]) -> float:
        """Calcula score de calidad simplificado"""
        score = 0.0
        
        # Puntos por manos detectadas
        if landmarks['left_hand']:
            score += 0.4
        if landmarks['right_hand']:
            score += 0.4
        if landmarks['pose']:
            score += 0.2
            
        return min(score, 1.0)

    def _empty_result(self) -> Dict[str, Any]:
        """Resultado vacío para errores"""
        return {
            'results': None,
            'landmarks': {},
            'quality_score': 0.0,
            'is_valid': False,
            'timestamp': datetime.now().isoformat(),
            'has_motion_data': False,
            'processing_time_ms': 0.0
        }

    def close(self):
        """Libera recursos"""
        if self._is_closed:
            return
            
        try:
            with self._lock:
                if self.holistic:
                    self.holistic.close()
                    self.holistic = None
                self.motion_history.clear()
                self._is_closed = True
                logger.info("✅ Gesture extractor closed")
        except Exception as e:
            logger.error(f"Error closing extractor: {e}")

# ===========================
# RECONOCEDOR CORREGIDO - FUNCIONAL
# ===========================
class FunctionalGestureRecognizer:
    def __init__(self):
        self.similarity_threshold = 0.8  # AUMENTADO para mejor matching
        self.min_confidence = 0.6        # REDUCIDO para más detecciones
        self.max_frames_compare = 8
        
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normaliza puntos"""
        if points.shape[0] == 0:
            return points
            
        center = np.mean(points, axis=0)
        normalized = points - center
        
        scale = np.mean(np.linalg.norm(normalized, axis=1))
        if scale > 1e-8:
            normalized /= scale
            
        return normalized

    def extract_features_simple(self, landmarks_dict: Dict[str, Any]) -> List[float]:
        """Extrae características SIMPLIFICADAS y efectivas"""
        features: List[float] = []
        
        # 1. Características básicas de manos
        for hand_type in ['left_hand', 'right_hand']:
            hand_data = landmarks_dict.get(hand_type, [])
            if len(hand_data) == 63:  # 21 puntos * 3
                points = np.array(hand_data).reshape(-1, 3)
                
                # Distancias clave entre puntos de la mano
                key_distances = [
                    (0, 1),   # Muñeca a base del índice
                    (0, 5),   # Muñeca a base del meñique
                    (5, 9),   # Base meñique a base anular
                    (9, 13),  # Base anular a base medio
                    (13, 17), # Base medio a base índice
                    (8, 12),  # Punta índice a punta medio
                ]
                
                for idx1, idx2 in key_distances:
                    if idx1 < len(points) and idx2 < len(points):
                        dist = np.linalg.norm(points[idx1] - points[idx2])
                        features.append(float(dist))
        
        # 2. Características de pose simplificadas
        pose_data = landmarks_dict.get('pose', [])
        if len(pose_data) >= 9 * 3:
            points = np.array(pose_data).reshape(-1, 3)
            
            # Distancias entre hombros y muñecas
            if len(points) >= 7:
                # Hombro izquierdo (1) a hombro derecho (2)
                dist_shoulders = np.linalg.norm(points[1] - points[2])
                features.append(float(dist_shoulders))
                
                # Hombro izquierdo a muñeca izquierda (5)
                dist_left_arm = np.linalg.norm(points[1] - points[5])
                features.append(float(dist_left_arm))
                
                # Hombro derecho a muñeca derecha (6)
                dist_right_arm = np.linalg.norm(points[2] - points[6])
                features.append(float(dist_right_arm))
        
        # 3. Características de movimiento básicas
        motion_vectors = landmarks_dict.get('motion_vectors', {})
        for point in ['left_wrist', 'right_wrist']:
            vector = motion_vectors.get(point, {})
            features.append(float(vector.get('velocity', 0.0)))
            features.append(float(vector.get('magnitude', 0.0)))
        
        # Rellenar y limitar características
        while len(features) < 20:
            features.append(0.0)
            
        return features[:20]  # Máximo 20 características

    def calculate_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calcula similitud con algoritmo robusto"""
        if not features1 or not features2:
            return float('inf')
            
        feat1, feat2 = np.array(features1, dtype=np.float32), np.array(features2, dtype=np.float32)
        min_len = min(len(feat1), len(feat2))
        
        if min_len == 0:
            return float('inf')
        
        # Usar distancia coseno para mejor robustez
        feat1_norm = feat1[:min_len] / (np.linalg.norm(feat1[:min_len]) + 1e-8)
        feat2_norm = feat2[:min_len] / (np.linalg.norm(feat2[:min_len]) + 1e-8)
        
        cosine_sim = np.dot(feat1_norm, feat2_norm)
        distance = 1.0 - cosine_sim  # Convertir similitud a distancia
        
        return float(distance)

    def find_best_match(self, input_sequence: List[List[float]], 
                       stored_gestures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Encuentra la mejor coincidencia - ALGORITMO CORREGIDO"""
        if not input_sequence or not stored_gestures:
            logger.info("❌ No hay secuencia de entrada o gestos almacenados")
            return None
            
        best_match = None
        best_confidence = 0.0
        input_features = input_sequence[:self.max_frames_compare]
        
        logger.info(f"🔍 Buscando entre {len(stored_gestures)} gestos...")
        
        for gesture in stored_gestures:
            gesture_name = gesture.get("name", "Desconocido")
            stored_features = gesture.get('features_sequence', [])
            
            if not stored_features:
                continue
            
            stored_features = stored_features[:self.max_frames_compare]
            min_frames = min(len(input_features), len(stored_features))
            
            if min_frames < 2:
                continue
                
            total_similarity = 0.0
            valid_comparisons = 0
            
            # Comparar frames alineados
            for i in range(min_frames):
                distance = self.calculate_similarity(input_features[i], stored_features[i])
                if distance < float('inf'):
                    similarity_score = 1.0 - distance  # Convertir distancia a similitud
                    total_similarity += similarity_score
                    valid_comparisons += 1
            
            if valid_comparisons > 0:
                avg_similarity = total_similarity / valid_comparisons
                confidence = avg_similarity  # La similitud es directamente la confianza
                
                logger.info(f"📊 Gesto '{gesture_name}': similitud={avg_similarity:.3f}, confianza={confidence:.3f}")
                
                if confidence > best_confidence and confidence >= self.min_confidence:
                    best_confidence = confidence
                    best_match = {
                        "id": str(gesture["_id"]),
                        "name": gesture_name,
                        "description": gesture.get("description", ""),
                        "video_path": gesture.get("video_path", ""),
                        "similarity": float(avg_similarity),
                        "confidence": float(confidence),
                        "motion_aware": True,
                        "processing_mode": "FUNCTIONAL_RECOGNITION"
                    }
        
        if best_match:
            logger.info(f"🎯 MEJOR COINCIDENCIA: {best_match['name']} (confianza: {best_match['confidence']:.3f})")
        else:
            logger.info("❌ No se encontró ninguna coincidencia con confianza suficiente")
            
        return best_match

# Instancias globales
_extractor: Optional[HighPrecisionGestureExtractor] = None
_recognizer = FunctionalGestureRecognizer()

# ===========================
# SÍNTESIS DE AUDIO
# ===========================
def text_to_speech_enhanced(text: str, lang: str = 'es') -> Optional[str]:
    """Convierte texto a audio"""
    try:
        if not text or not text.strip():
            return None
            
        tts = gTTS(text=text.strip(), lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_b64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        return f"data:audio/mp3;base64,{audio_b64}"
    except Exception as e:
        logger.error(f"❌ Error en síntesis de audio: {e}")
        return None

# ===========================
# ENDPOINTS PRINCIPALES CORREGIDOS
# ===========================

@gestures_bp.route('/uploads/gestures/<filename>')
@login_required
def serve_gesture_video(filename: str):
    """Sirve archivos de video de gestos"""
    try:
        if '..' in filename or filename.startswith('/'):
            return "Acceso denegado", 403
            
        uploads_dir = os.path.join(current_app.static_folder, 'uploads', 'gestures')
        return send_from_directory(uploads_dir, filename)
    except Exception as e:
        logger.error(f"❌ Error sirviendo video {filename}: {e}")
        return "Video no encontrado", 404

@gestures_bp.route("/gestures")
@login_required
def dashboard():
    """Dashboard principal"""
    return render_template("dashboard.html")

@gestures_bp.route("/api/process_frame", methods=["POST"])
@login_required
def process_frame():
    """Procesamiento en tiempo real"""
    try:
        global _extractor
        if _extractor is None:
            return jsonify({"success": False, "error": "Sistema no inicializado"}), 500
            
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Datos JSON inválidos"}), 400
            
        frame_data = data.get('image')
        if not frame_data:
            return jsonify({"success": False, "error": "Frame requerido"}), 400
        
        img = decode_datauri_to_bgr(frame_data)
        if img is None:
            return jsonify({"success": False, "error": "Frame inválido"}), 400
        
        img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
        result = _extractor.process_frame(img)
        
        annotated_uri = None
        if result['results']:
            annotated_img = _visualizer.draw_landmarks(img, result['results'])
            annotated_uri = bgr_to_datauri(annotated_img)
        
        response_data = {
            "success": True,
            "annotated_frame": annotated_uri,
            "landmarks_detected": {
                "left_hand": len(result['landmarks'].get('left_hand', [])) > 0,
                "right_hand": len(result['landmarks'].get('right_hand', [])) > 0,
                "pose": len(result['landmarks'].get('pose', [])) > 0,
            },
            "quality_score": result['quality_score'],
            "is_valid": result['is_valid'],
            "has_motion_data": result.get('has_motion_data', False),
            "processing_time_ms": result.get('processing_time_ms', 0)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Error en process_frame: {str(e)}")
        return jsonify({"success": False, "error": "Error procesando frame"}), 500

@gestures_bp.route("/api/register_gesture", methods=["POST"])
@login_required
def register_gesture():
    """Registro de gestos - VERSIÓN CORREGIDA CON LANDMARKS"""
    try:
        logger.info("🎯 Registro de gesto...")
        
        global _extractor, _recognizer
        if _extractor is None:
            return jsonify({"success": False, "error": "Sistema no inicializado"}), 500
            
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Datos JSON inválidos"}), 400
            
        name = (data.get('name') or "").strip()
        frames = data.get('frames', [])
        description = data.get('description', '')
        category = data.get('category', 'general')
        
        logger.info(f"📝 Registrando: {name}, frames: {len(frames)}")
        
        if not name:
            return jsonify({"success": False, "error": "Nombre requerido"}), 400
        
        if len(frames) < MIN_VALID_FRAMES_TO_REGISTER:
            return jsonify({"success": False, "error": f"Mínimo {MIN_VALID_FRAMES_TO_REGISTER} frames"}), 400
        
        features_sequence = []
        valid_frames = 0
        video_frames = []
        total_processing_time = 0
        
        # NUEVO: Contadores para landmarks
        landmarks_stats = {
            'left_hand_frames': 0,
            'right_hand_frames': 0,
            'pose_frames': 0,
            'face_frames': 0
        }
        
        for frame_data in frames[:MAX_SAVE_FRAMES]:
            img = decode_datauri_to_bgr(frame_data)
            if img is None:
                continue
                
            img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
            result = _extractor.process_frame(img)
            total_processing_time += result.get('processing_time_ms', 0)
            
            if result['is_valid'] and result['landmarks']:
                # Usar el extractor simplificado para registro también
                features = _recognizer.extract_features_simple(result['landmarks'])
                features_sequence.append(features)
                valid_frames += 1
                
                # NUEVO: Contar landmarks detectados en este frame
                landmarks = result['landmarks']
                if landmarks.get('left_hand'):
                    landmarks_stats['left_hand_frames'] += 1
                if landmarks.get('right_hand'):
                    landmarks_stats['right_hand_frames'] += 1
                if landmarks.get('pose'):
                    landmarks_stats['pose_frames'] += 1
                # Nota: Face no se está detectando en el extractor actual
                
                if result['results']:
                    annotated_img = _visualizer.draw_landmarks(img, result['results'])
                    video_frames.append(annotated_img)
        
        logger.info(f"✅ Frames procesados: {valid_frames}/{len(frames)}")
        logger.info(f"📊 Landmarks stats: {landmarks_stats}")
        
        if valid_frames < MIN_VALID_FRAMES_TO_REGISTER:
            return jsonify({"success": False, "error": "Frames válidos insuficientes"}), 400
        
        video_path = generate_gesture_video(video_frames, name) if video_frames else None
        
        try:
            mongo = current_app.mongo
            
            # NUEVO: Calcular landmarks_count como porcentaje de frames con detección
            landmarks_count = {
                "left_hand": int((landmarks_stats['left_hand_frames'] / valid_frames) * 100) if valid_frames > 0 else 0,
                "right_hand": int((landmarks_stats['right_hand_frames'] / valid_frames) * 100) if valid_frames > 0 else 0,
                "pose": int((landmarks_stats['pose_frames'] / valid_frames) * 100) if valid_frames > 0 else 0,
                "face": 0  # Por ahora no detectamos cara
            }
            
            gesture_doc = {
                "_id": ObjectId(),
                "name": name,
                "description": description,
                "category": category,
                "features_sequence": features_sequence,
                "total_frames": len(features_sequence),
                "valid_frames": valid_frames,
                "video_path": video_path,
                "created_by": str(current_user.id),
                "created_at": datetime.utcnow(),
                "avg_quality": 0.8,
                "has_motion_data": True,
                "landmarks_count": landmarks_count,  # NUEVO: Guardar landmarks
                "landmarks_stats": landmarks_stats   # NUEVO: Stats detallados para debug
            }
            
            mongo.db.gestures.insert_one(gesture_doc)
            
            return jsonify({
                "success": True,
                "gesture_id": str(gesture_doc["_id"]),
                "name": name,
                "frames_registered": valid_frames,
                "video_path": video_path,
                "has_motion_data": True,
                "landmarks_count": landmarks_count,  # NUEVO: Incluir en respuesta
                "avg_processing_time_ms": float(total_processing_time / valid_frames) if valid_frames > 0 else 0,
                "message": "Gesto registrado exitosamente"
            })
            
        except Exception as db_error:
            logger.error(f"❌ Error DB: {db_error}")
            return jsonify({"success": False, "error": "Error guardando en base de datos"}), 500
            
    except Exception as e:
        logger.error(f"❌ Error en register_gesture: {str(e)}")
        return jsonify({"success": False, "error": "Error registrando gesto"}), 500

@gestures_bp.route("/api/recognize_gesture", methods=["POST"])
@login_required
def recognize_gesture():
    """Reconocimiento de gestos - VERSIÓN FUNCIONAL"""
    try:
        start_time = time.perf_counter()
        
        global _extractor, _recognizer
        if _extractor is None:
            return jsonify({"success": False, "error": "Sistema no inicializado"}), 500
            
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Datos JSON inválidos"}), 400
            
        frames = data.get('frames', [])
        
        logger.info(f"🚀 Reconocimiento con {len(frames)} frames")
        
        if not frames:
            return jsonify({"success": False, "error": "Frames requeridos"}), 400
        
        input_features_sequence = []
        valid_frames = 0
        total_processing_time = 0
        
        for frame_data in frames[:_recognizer.max_frames_compare]:
            img = decode_datauri_to_bgr(frame_data)
            if img is None:
                continue
                
            img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
            result = _extractor.process_frame(img)
            total_processing_time += result.get('processing_time_ms', 0)
            
            if result['landmarks'] and result['is_valid']:
                features = _recognizer.extract_features_simple(result['landmarks'])
                input_features_sequence.append(features)
                valid_frames += 1
        
        logger.info(f"✅ {valid_frames} frames procesados en {total_processing_time:.1f}ms")
        
        if not input_features_sequence:
            return jsonify({"success": False, "error": "No se pudieron extraer características"}), 400
        
        best_match = None
        audio_data = None
        
        try:
            mongo = current_app.mongo
            stored_gestures = list(mongo.db.gestures.find(
                {"created_by": str(current_user.id)},
                {"name": 1, "description": 1, "video_path": 1, "features_sequence": 1}
            ))
            
            logger.info(f"📚 Buscando en {len(stored_gestures)} gestos almacenados")
            
            if stored_gestures:
                best_match = _recognizer.find_best_match(input_features_sequence, stored_gestures)
                
                total_time = (time.perf_counter() - start_time) * 1000
                
                if best_match:
                    audio_data = text_to_speech_enhanced(f"Reconocido: {best_match['name']}")
                    
                    return jsonify({
                        "success": True,
                        "recognized_gesture": best_match,
                        "frames_processed": len(input_features_sequence),
                        "audio_data": audio_data,
                        "processing_time_ms": float(total_time),
                        "avg_frame_time_ms": float(total_processing_time / valid_frames) if valid_frames > 0 else 0,
                        "message": f"Gesto '{best_match['name']}' reconocido con confianza {best_match['confidence']:.3f}"
                    })
                else:
                    return jsonify({
                        "success": True,
                        "recognized_gesture": None,
                        "frames_processed": len(input_features_sequence),
                        "processing_time_ms": float(total_time),
                        "message": "No se reconoció ningún gesto con suficiente confianza"
                    })
            else:
                total_time = (time.perf_counter() - start_time) * 1000
                return jsonify({
                    "success": True,
                    "recognized_gesture": None,
                    "frames_processed": len(input_features_sequence),
                    "processing_time_ms": float(total_time),
                    "message": "No hay gestos almacenados para comparar"
                })
                
        except Exception as db_error:
            logger.error(f"❌ Error en BD: {db_error}")
            return jsonify({"success": False, "error": "Error accediendo a la base de datos"}), 500
            
    except Exception as e:
        logger.error(f"❌ Error en recognize_gesture: {str(e)}")
        return jsonify({"success": False, "error": f"Error en reconocimiento: {str(e)}"}), 500

# ===========================
# ENDPOINTS DE GESTIÓN
# ===========================

@gestures_bp.route("/api/gestures_list", methods=["GET"])
@login_required
def get_gestures_list():
    """Obtiene lista de gestos del usuario"""
    try:
        mongo = current_app.mongo
        user_gestures = list(mongo.db.gestures.find(
            {"created_by": str(current_user.id)},
            {"name": 1, "description": 1, "category": 1, "created_at": 1, 
             "total_frames": 1, "valid_frames": 1, "video_path": 1, "avg_quality": 1, "has_motion_data": 1}
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
                "avg_quality": float(gesture.get("avg_quality", 0)),
                "has_motion_data": bool(gesture.get("has_motion_data", False)),
                "created_at": gesture.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")
            })
        
        return jsonify({"success": True, "gestures": gestures_data})
        
    except Exception as e:
        logger.error(f"❌ Error en get_gestures_list: {str(e)}")
        return jsonify({"success": False, "error": "Error obteniendo gestos"}), 500

@gestures_bp.route("/api/gesture_details/<gesture_id>", methods=["GET"])
@login_required
def get_gesture_details(gesture_id):
    """Obtiene detalles de un gesto específico - VERSIÓN MEJORADA"""
    try:
        mongo = current_app.mongo
        gesture = mongo.db.gestures.find_one({
            "_id": ObjectId(gesture_id),
            "created_by": str(current_user.id)
        })
        
        if not gesture:
            return jsonify({"success": False, "error": "Gesto no encontrado"}), 404
        
        # MEJORADO: Manejar todos los campos posibles
        features_metadata = {
            "features_per_frame": len(gesture.get('features_sequence', [[]])[0]) if gesture.get('features_sequence') and len(gesture['features_sequence']) > 0 else 0,
            "total_frames_in_sequence": len(gesture.get('features_sequence', []))
        }
        
        # MEJORADO: Proporcionar valores por defecto robustos
        landmarks_count = gesture.get('landmarks_count', {})
        landmarks_stats = gesture.get('landmarks_stats', {})
        
        gesture_data = {
            "id": str(gesture["_id"]),
            "name": gesture.get("name", ""),
            "description": gesture.get("description", ""),
            "category": gesture.get("category", "general"),
            "total_frames": gesture.get("total_frames", 0),
            "valid_frames": gesture.get("valid_frames", 0),
            "video_path": gesture.get("video_path", ""),
            "avg_quality": float(gesture.get("avg_quality", 0)),
            "has_motion_data": bool(gesture.get("has_motion_data", False)),
            "created_at": gesture.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M"),
            "features_metadata": features_metadata,
            "landmarks_count": {
                "left_hand": landmarks_count.get("left_hand", 0),
                "right_hand": landmarks_count.get("right_hand", 0),
                "pose": landmarks_count.get("pose", 0),
                "face": landmarks_count.get("face", 0)
            },
            "landmarks_stats": landmarks_stats  # NUEVO: Incluir stats detallados
        }
        
        logger.info(f"📊 Enviando detalles del gesto: {gesture_data['name']}, landmarks: {gesture_data['landmarks_count']}")
        
        return jsonify({"success": True, "gesture": gesture_data})
        
    except Exception as e:
        logger.error(f"❌ Error en get_gesture_details: {str(e)}")
        return jsonify({"success": False, "error": f"Error obteniendo detalles: {str(e)}"}), 500
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
                "avg_quality": float(gesture.get("avg_quality", 0)),
                "has_motion_data": bool(gesture.get("has_motion_data", False))
            })
        
        return jsonify({
            "success": True,
            "query": query,
            "gestures": results,
            "total_matches": len(results)
        })
        
    except Exception as e:
        logger.error(f"❌ Error en search_gestures: {str(e)}")
        return jsonify({"success": False, "error": "Error buscando gestos"}), 500

def _calculate_relevance_score(gesture: Dict[str, Any], search_words: List[str]) -> int:
    """Calcula puntuación de relevancia"""
    name = gesture.get('name', '').lower()
    description = gesture.get('description', '').lower()
    
    score = 0
    for word in search_words:
        if word in name:
            score += 5
        if word in description:
            score += 1
    
    return score

@gestures_bp.route("/api/search_gestures_phrase", methods=["POST"])
@login_required
def search_gestures_phrase():
    """Busca gestos por frase compuesta"""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Datos JSON inválidos"}), 400
            
        phrase = data.get('phrase', '').strip().lower()
        
        if not phrase:
            return jsonify({"success": False, "error": "Frase requerida"}), 400
        
        words = re.findall(r'\b\w+\b', phrase)
        if not words:
            return jsonify({"success": False, "error": "No se encontraron palabras"}), 400
        
        mongo = current_app.mongo
        user_id = str(current_user.id)
        
        # Búsqueda optimizada
        exact_phrase_gestures = list(mongo.db.gestures.find({
            "created_by": user_id,
            "$or": [
                {"name": {"$regex": f"^{phrase}$", "$options": "i"}},
                {"name": {"$regex": f"\\b{phrase}\\b", "$options": "i"}}
            ]
        }))
        
        if not exact_phrase_gestures:
            # Búsqueda por todas las palabras
            and_conditions = [{
                "$or": [
                    {"name": {"$regex": word, "$options": "i"}},
                    {"description": {"$regex": word, "$options": "i"}}
                ]
            } for word in words]
            
            all_words_gestures = list(mongo.db.gestures.find({
                "created_by": user_id,
                "$and": and_conditions
            }))
            
            if not all_words_gestures:
                # Búsqueda por alguna palabra
                or_conditions = []
                for word in words:
                    or_conditions.extend([
                        {"name": {"$regex": word, "$options": "i"}},
                        {"description": {"$regex": word, "$options": "i"}}
                    ])
                
                some_words_gestures = list(mongo.db.gestures.find({
                    "created_by": user_id,
                    "$or": or_conditions
                }))
                
                some_words_gestures.sort(key=lambda x: _calculate_relevance_score(x, words), reverse=True)
                matching_gestures = some_words_gestures
            else:
                matching_gestures = all_words_gestures
        else:
            matching_gestures = exact_phrase_gestures
        
        # Procesar resultados
        results = []
        for gesture in matching_gestures:
            relevance_score = _calculate_relevance_score(gesture, words)
            
            results.append({
                "id": str(gesture["_id"]),
                "name": gesture.get("name", ""),
                "description": gesture.get("description", ""),
                "category": gesture.get("category", "general"),
                "video_path": gesture.get("video_path", ""),
                "avg_quality": float(gesture.get("avg_quality", 0)),
                "has_motion_data": bool(gesture.get("has_motion_data", False)),
                "relevance_score": relevance_score,
                "matches_phrase": phrase.lower() in gesture.get('name', '').lower()
            })
        
        results.sort(key=lambda x: (x['matches_phrase'], x['relevance_score']), reverse=True)
        
        return jsonify({
            "success": True,
            "phrase": phrase,
            "words_found": words,
            "gestures_found": results,
            "total_matches": len(results),
            "search_type": "exact_phrase" if exact_phrase_gestures else "all_words" if 'all_words_gestures' in locals() and all_words_gestures else "some_words"
        })
        
    except Exception as e:
        logger.error(f"❌ Error en search_gestures_phrase: {str(e)}")
        return jsonify({"success": False, "error": "Error buscando gestos"}), 500

@gestures_bp.route("/api/delete_gesture/<gesture_id>", methods=["DELETE"])
@login_required
def delete_gesture(gesture_id):
    """Elimina un gesto"""
    try:
        mongo = current_app.mongo
        
        gesture = mongo.db.gestures.find_one({
            "_id": ObjectId(gesture_id),
            "created_by": str(current_user.id)
        })
        
        if not gesture:
            return jsonify({"success": False, "error": "Gesto no encontrado"}), 404
        
        # Eliminar video asociado
        video_path = gesture.get("video_path")
        if video_path:
            try:
                full_path = os.path.join(current_app.static_folder, video_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    logger.info(f"✅ Video eliminado: {video_path}")
            except Exception as video_error:
                logger.warning(f"⚠️ No se pudo eliminar video: {video_error}")
        
        result = mongo.db.gestures.delete_one({"_id": ObjectId(gesture_id)})
        
        if result.deleted_count > 0:
            return jsonify({"success": True, "message": "Gesto eliminado exitosamente"})
        else:
            return jsonify({"success": False, "error": "Error eliminando gesto"}), 500
            
    except Exception as e:
        logger.error(f"❌ Error en delete_gesture: {str(e)}")
        return jsonify({"success": False, "error": "Error eliminando gesto"}), 500

@gestures_bp.route("/api/gesture_stats", methods=["GET"])
@login_required
def get_gesture_stats():
    """Obtiene estadísticas de gestos"""
    try:
        mongo = current_app.mongo
        
        total_gestures = mongo.db.gestures.count_documents({"created_by": str(current_user.id)})
        
        gestures_with_video = mongo.db.gestures.count_documents({
            "created_by": str(current_user.id),
            "video_path": {"$exists": True, "$ne": ""}
        })
        
        gestures_with_motion = mongo.db.gestures.count_documents({
            "created_by": str(current_user.id),
            "has_motion_data": True
        })
        
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
                "gestures_with_motion": gestures_with_motion
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error en get_gesture_stats: {str(e)}")
        return jsonify({"success": False, "error": "Error obteniendo estadísticas"}), 500

@gestures_bp.route("/api/system_status", methods=["GET"])
@login_required
def get_system_status():
    """Obtiene estado del sistema"""
    try:
        extractor_status = "active" if _extractor and not _extractor._is_closed else "inactive"
        
        mongo_status = "connected"
        try:
            mongo = current_app.mongo
            mongo.db.command('ping')
        except Exception:
            mongo_status = "disconnected"
        
        return jsonify({
            "success": True,
            "details": {
                "extractor": extractor_status,
                "database": mongo_status,
                "recognition_mode": "FUNCTIONAL_RECOGNITION",
                "processing_optimized": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error en get_system_status: {str(e)}")
        return jsonify({"success": False, "error": "Error obteniendo estado"}), 500

@gestures_bp.route("/api/text_to_speech", methods=["POST"])
@login_required
def api_text_to_speech():
    """Endpoint para síntesis de voz"""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Datos JSON inválidos"}), 400
            
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"success": False, "error": "Texto requerido"}), 400
        
        audio_data = text_to_speech_enhanced(text)
        
        if audio_data:
            return jsonify({
                "success": True,
                "audio_data": audio_data
            })
        else:
            return jsonify({"success": False, "error": "Error generando audio"}), 500
            
    except Exception as e:
        logger.error(f"❌ Error en text_to_speech: {str(e)}")
        return jsonify({"success": False, "error": "Error en síntesis"}), 500

@gestures_bp.route("/api/reinitialize_extractor", methods=["POST"])
@login_required
def reinitialize_extractor():
    """Reinicializa el extractor"""
    try:
        global _extractor
        
        if _extractor:
            _extractor.close()
        
        _extractor = HighPrecisionGestureExtractor(current_app)
        
        return jsonify({
            "success": True,
            "message": "Extractor reinicializado correctamente"
        })
        
    except Exception as e:
        logger.error(f"❌ Error en reinitialize_extractor: {str(e)}")
        return jsonify({"success": False, "error": "Error reinicializando"}), 500

@gestures_bp.route("/api/motion_analysis", methods=["POST"])
@login_required
def analyze_motion():
    """Analiza características de movimiento"""
    try:
        if _extractor is None:
            return jsonify({"success": False, "error": "Sistema no inicializado"}), 500
            
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Datos JSON inválidos"}), 400
            
        frame_data = data.get('image')
        if not frame_data:
            return jsonify({"success": False, "error": "Frame requerido"}), 400
        
        img = decode_datauri_to_bgr(frame_data)
        if img is None:
            return jsonify({"success": False, "error": "Frame inválido"}), 400
        
        img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
        result = _extractor.process_frame(img)
        
        motion_analysis = {
            "has_motion_data": result.get('has_motion_data', False),
            "motion_vectors": result['landmarks'].get('motion_vectors', {}),
            "motion_history_length": len(_extractor.motion_history)
        }
        
        return jsonify({
            "success": True,
            "motion_analysis": motion_analysis,
            "quality_score": result['quality_score']
        })
        
    except Exception as e:
        logger.error(f"❌ Error en motion_analysis: {str(e)}")
        return jsonify({"success": False, "error": "Error analizando movimiento"}), 500

# ===========================
# INICIALIZACIÓN
# ===========================
def init_extractor(app: Any) -> bool:
    """Inicializa el extractor"""
    global _extractor
    try:
        _extractor = HighPrecisionGestureExtractor(app)
        app.logger.info("✅ Gesture Extractor initialized")
        return True
    except Exception as e:
        app.logger.error(f"❌ Error initializing extractor: {e}")
        _extractor = None
        return False

def close_extractor():
    """Cierra el extractor"""
    global _extractor
    if _extractor:
        _extractor.close()
        _extractor = None

def init_gestures_module(app: Any) -> bool:
    """Inicializa el módulo de gestos"""
    try:
        success = init_extractor(app)
        if not success:
            app.logger.error("❌ Failed to initialize gestures module")
            return False
        
        ensure_upload_dirs()
        
        app.logger.info("✅ Gestures module initialized - FUNCTIONAL MODE")
        return True
        
    except Exception as e:
        app.logger.error(f"❌ Error initializing gestures module: {e}")
        return False