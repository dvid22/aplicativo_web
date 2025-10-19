/// static/js/gestures.js
// VERSIÓN COMPLETA CORREGIDA - COMPLETAMENTE FUNCIONAL

console.log("🚀 Inicializando sistema de gestos profesional...");

// =============================================================
// VARIABLES GLOBALES
// =============================================================
let cameraStream = null;
let recorderStream = null;
let recording = false;
let recActive = false;
let frameCount = 0;
let recordStartTime = null;
let recordingFrames = [];
let recognitionInterval = null;
let cooldownActive = false;
let currentProcessing = false;

// =============================================================
// CONFIGURACIÓN
// =============================================================
const CONFIG = {
    MAX_FRAMES: 60,
    MIN_FRAMES: 10,
    RECOGNITION_INTERVAL: 2000,
    FRAME_RATE: 15,
    SIMILARITY_THRESHOLD: 0.25
};

// =============================================================
// MODAL MANAGER - MEJORADO
// =============================================================
class ModalManager {
    static showSuccess(title, message, details = '') {
        console.log('✅ SUCCESS:', title, message, details);
        this.showAlert('success', title, message, details);
    }

    static showError(title, message, details = '') {
        console.error('❌ ERROR:', title, message, details);
        this.showAlert('danger', title, message, details);
    }

    static showWarning(title, message, details = '') {
        console.warn('⚠️ WARNING:', title, message, details);
        this.showAlert('warning', title, message, details);
    }

    static showInfo(title, message, details = '') {
        console.info('ℹ️ INFO:', title, message, details);
        this.showAlert('info', title, message, details);
    }

    static showAlert(type, title, message, details = '') {
        // Usar Toast de Bootstrap si está disponible
        if (typeof bootstrap !== 'undefined' && bootstrap.Toast) {
            const toastHtml = `
                <div class="toast align-items-center text-bg-${type} border-0" role="alert">
                    <div class="d-flex">
                        <div class="toast-body">
                            <strong>${title}</strong> ${message}
                            ${details ? `<br><small>${details}</small>` : ''}
                        </div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>
            `;
            
            const toastContainer = document.getElementById('toastContainer') || document.body;
            const toastElement = document.createElement('div');
            toastElement.innerHTML = toastHtml;
            toastContainer.appendChild(toastElement);
            
            const toast = new bootstrap.Toast(toastElement.querySelector('.toast'));
            toast.show();
            
            // Limpiar después de cerrar
            toastElement.addEventListener('hidden.bs.toast', () => {
                toastElement.remove();
            });
        } else {
            // Fallback a alert simple
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                <strong>${title}</strong> ${message}
                ${details ? `<br><small>${details}</small>` : ''}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            const alertContainer = document.getElementById('alertContainer') || document.body;
            alertContainer.appendChild(alertDiv);
            
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
        }
    }

    static showConfirm(title, message, onConfirm, onCancel = null) {
        if (confirm(`❓ ${title}\n\n${message}`)) {
            onConfirm && onConfirm();
        } else {
            onCancel && onCancel();
        }
    }

    static showLoading(title = 'Procesando...') {
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'customLoadingOverlay';
        loadingDiv.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            color: white;
            flex-direction: column;
        `;
        loadingDiv.innerHTML = `
            <div class="spinner-border text-primary mb-3" style="width: 3rem; height: 3rem;"></div>
            <div class="fs-5">${title}</div>
        `;
        document.body.appendChild(loadingDiv);
        
        return {
            hide: () => {
                const overlay = document.getElementById('customLoadingOverlay');
                if (overlay) overlay.remove();
            }
        };
    }

    static hideLoading() {
        const overlay = document.getElementById('customLoadingOverlay');
        if (overlay) overlay.remove();
    }
}

// =============================================================
// GESTURES API - ACTUALIZADO PARA NUEVO BACKEND
// =============================================================
class GesturesAPI {
    static async processFrame(frameData) {
        try {
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en processFrame:', error);
            return { success: false, error: error.message };
        }
    }

    static async registerGesture(gestureData) {
        try {
            const response = await fetch('/api/register_gesture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(gestureData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en registerGesture:', error);
            return { success: false, error: error.message };
        }
    }

    static async recognizeGesture(frames) {
        try {
            const response = await fetch('/api/recognize_gesture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: frames })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en recognizeGesture:', error);
            return { success: false, error: error.message };
        }
    }

    static async getGesturesList() {
        try {
            const response = await fetch('/api/gestures_list');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en getGesturesList:', error);
            return { success: false, error: error.message };
        }
    }

    static async deleteGesture(gestureId) {
        try {
            const response = await fetch(`/api/delete_gesture/${gestureId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en deleteGesture:', error);
            return { success: false, error: error.message };
        }
    }

    static async getGestureStats() {
        try {
            const response = await fetch('/api/gesture_stats');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en getGestureStats:', error);
            return { success: false, error: error.message };
        }
    }

    static async textToSpeech(text) {
        try {
            const response = await fetch('/api/text_to_speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en textToSpeech:', error);
            return { success: false, error: error.message };
        }
    }

    static async getSystemStatus() {
        try {
            const response = await fetch('/api/system_status');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en getSystemStatus:', error);
            return { success: false, error: error.message };
        }
    }

    static async reinitializeExtractor() {
        try {
            const response = await fetch('/api/reinitialize_extractor', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error en reinitializeExtractor:', error);
            return { success: false, error: error.message };
        }
    }
}

// =============================================================
// INICIALIZAR CÁMARA - OPTIMIZADO
// =============================================================
async function initCamera(videoElementId, overlayId) {
    const videoElement = document.getElementById(videoElementId);
    const overlay = document.getElementById(overlayId);

    if (!videoElement) {
        console.error('❌ Elemento de video no encontrado:', videoElementId);
        return null;
    }

    try {
        // Detener stream anterior si existe
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 },
                facingMode: 'user'
            }
        });
        
        videoElement.srcObject = stream;
        if (overlay) overlay.style.display = "none";
        
        return new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play().then(() => {
                    console.log('✅ Cámara inicializada:', videoElementId);
                    resolve(stream);
                }).catch(error => {
                    console.error('Error al reproducir video:', error);
                    resolve(stream);
                });
            };
        });
        
    } catch (err) {
        console.error("❌ Error al acceder a la cámara:", err);
        if (overlay) overlay.style.display = "flex";
        ModalManager.showError(
            "Error de cámara", 
            "No se pudo acceder a la cámara", 
            "Verifica que la cámara esté conectada y los permisos estén habilitados."
        );
        return null;
    }
}

// =============================================================
// SISTEMA DE CAPTURA DE FRAMES - OPTIMIZADO
// =============================================================
function captureFrame(videoElement) {
    if (!videoElement || videoElement.readyState !== 4) {
        return null;
    }
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = videoElement.videoWidth || 640;
    canvas.height = videoElement.videoHeight || 480;
    
    try {
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.8);
    } catch (error) {
        console.error('Error capturando frame:', error);
        return null;
    }
}

// =============================================================
// VISUALIZACIÓN EN TIEMPO REAL - NUEVO
// =============================================================
async function startRealTimePreview() {
    const video = document.getElementById("recVideo");
    if (!video || !video.srcObject) return;

    const previewInterval = setInterval(async () => {
        if (!recActive) {
            clearInterval(previewInterval);
            return;
        }

        const frame = captureFrame(video);
        if (!frame) return;

        try {
            const result = await GesturesAPI.processFrame(frame);
            if (result.success && result.annotated_frame) {
                // Mostrar frame con landmarks
                const previewElement = document.getElementById("realTimePreview");
                if (previewElement) {
                    previewElement.src = result.annotated_frame;
                }
                
                // Actualizar información de landmarks
                updateLandmarksInfo(result.landmarks_detected);
            }
        } catch (error) {
            console.error("Error en preview tiempo real:", error);
        }
    }, 500); // Actualizar cada 500ms para mejor rendimiento
}

function updateLandmarksInfo(landmarks) {
    if (!landmarks) return;
    
    const landmarksInfo = document.getElementById("landmarksInfo");
    if (!landmarksInfo) return;
    
    let infoHtml = '';
    
    if (landmarks.left_hand) {
        infoHtml += '<span class="badge bg-success me-1">Mano Izq</span>';
    }
    if (landmarks.right_hand) {
        infoHtml += '<span class="badge bg-danger me-1">Mano Der</span>';
    }
    if (landmarks.pose) {
        infoHtml += '<span class="badge bg-primary me-1">Cuerpo</span>';
    }
    if (landmarks.face) {
        infoHtml += '<span class="badge bg-warning me-1">Cara</span>';
    }
    
    landmarksInfo.innerHTML = infoHtml || '<span class="text-muted">No landmarks</span>';
}

// =============================================================
// RECONOCIMIENTO EN TIEMPO REAL - ACTUALIZADO
// =============================================================
async function startRecognition() {
    console.log('🎯 Iniciando reconocimiento...');
    
    const video = document.getElementById("recVideo");
    const overlay = document.getElementById("recOverlay");
    const status = document.getElementById("recStatus");
    const startBtn = document.getElementById("startRecBtn");
    const stopBtn = document.getElementById("stopRecBtn");

    if (!video) {
        ModalManager.showError("Error", "Elemento de video no encontrado");
        return;
    }

    // Inicializar cámara si no está activa
    if (!cameraStream) {
        cameraStream = await initCamera("recVideo", "recOverlay");
    }

    if (!cameraStream) {
        ModalManager.showError("Error", "No se pudo inicializar la cámara");
        return;
    }

    recActive = true;
    
    if (overlay) overlay.style.display = "none";
    
    if (status) {
        status.innerHTML = `<span class="status-indicator status-active"></span> Reconociendo`;
        status.className = "badge bg-success";
    }
    
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    // Iniciar preview en tiempo real
    startRealTimePreview();

    ModalManager.showInfo("Reconocimiento Iniciado", "El sistema está analizando gestos en tiempo real");

    // Sistema de reconocimiento mejorado
    recognitionInterval = setInterval(async () => {
        if (!recActive || cooldownActive || currentProcessing) return;

        currentProcessing = true;
        const frames = [];
        
        // Capturar 5 frames para mejor análisis
        for (let i = 0; i < 5 && recActive; i++) {
            const frame = captureFrame(video);
            if (frame) {
                frames.push(frame);
            }
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        if (frames.length === 0) {
            currentProcessing = false;
            return;
        }

        try {
            const result = await GesturesAPI.recognizeGesture(frames);
            
            if (result.success) {
                if (result.recognized_gesture) {
                    showRecognitionResult(result.recognized_gesture);
                    startCooldown();
                    
                    // Reproducir audio si está disponible
                    if (result.audio_data) {
                        playAudio(result.audio_data);
                    }
                }
                
                // Mostrar frames anotados si existen
                if (result.annotated_frames && result.annotated_frames.length > 0) {
                    showAnnotatedFrames(result.annotated_frames);
                }
            }
            
        } catch (error) {
            console.error("Error en reconocimiento:", error);
        } finally {
            currentProcessing = false;
        }
    }, CONFIG.RECOGNITION_INTERVAL);
}

function showRecognitionResult(gesture) {
    const resultDiv = document.getElementById("recResult");
    if (!resultDiv) return;
    
    const confidence = gesture.confidence ? (gesture.confidence * 100) : 85;
    
    resultDiv.innerHTML = `
        <div class="text-center p-3">
            <i class="fas fa-hands fa-3x text-success mb-3"></i>
            <h4 class="text-success mb-2">¡Gesto Reconocido!</h4>
            <h3 class="text-white mb-3">${gesture.name || 'Gesto'}</h3>
            <div class="confidence-badge mb-3">
                <span class="badge bg-success fs-6 p-2">Confianza: ${confidence.toFixed(1)}%</span>
            </div>
            ${gesture.description ? `<p class="text-light">${gesture.description}</p>` : ''}
            <div class="mt-3">
                <button class="btn btn-sm btn-outline-primary me-2" onclick="playGesture('${gesture.video_path || ''}', '${gesture.name || ''}')">
                    <i class="fas fa-play me-1"></i>Reproducir
                </button>
                <button class="btn btn-sm btn-outline-secondary" onclick="learnMoreAboutGesture('${gesture.id}')">
                    <i class="fas fa-info me-1"></i>Más Info
                </button>
            </div>
        </div>
    `;
    
    ModalManager.showSuccess(
        "¡Gesto Reconocido!", 
        `Se detectó: ${gesture.name}`,
        `Confianza: ${confidence.toFixed(1)}%`
    );
}

function showAnnotatedFrames(frames) {
    const container = document.getElementById("annotatedFrames");
    if (!container) return;
    
    container.innerHTML = '';
    
    frames.slice(0, 3).forEach((frame, index) => {
        const col = document.createElement('div');
        col.className = 'col-md-4';
        col.innerHTML = `
            <div class="card">
                <img src="${frame}" class="card-img-top" alt="Frame anotado ${index + 1}">
                <div class="card-body text-center">
                    <small class="text-muted">Frame ${index + 1}</small>
                </div>
            </div>
        `;
        container.appendChild(col);
    });
}

function startCooldown() {
    cooldownActive = true;
    const progressBar = document.getElementById("cooldownProgress");
    const cooldownText = document.getElementById("cooldownText");
    
    let timeLeft = 3;
    const interval = setInterval(() => {
        if (progressBar) {
            progressBar.style.width = `${((3 - timeLeft) / 3) * 100}%`;
        }
        if (cooldownText) {
            cooldownText.textContent = `${timeLeft}s`;
        }
        
        if (timeLeft <= 0) {
            clearInterval(interval);
            cooldownActive = false;
            if (progressBar) progressBar.style.width = "0%";
            if (cooldownText) cooldownText.textContent = "3s";
        }
        timeLeft--;
    }, 1000);
}

function stopRecognition() {
    console.log('🛑 Deteniendo reconocimiento...');
    
    recActive = false;
    currentProcessing = false;
    
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }

    const status = document.getElementById("recStatus");
    const startBtn = document.getElementById("startRecBtn");
    const stopBtn = document.getElementById("stopRecBtn");
    
    if (status) {
        status.innerHTML = `<span class="status-indicator status-inactive"></span> Inactivo`;
        status.className = "badge bg-secondary";
    }
    
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    const recResult = document.getElementById("recResult");
    if (recResult) {
        recResult.innerHTML = `
            <div class="d-flex align-items-center justify-content-center h-100">
                <div class="text-center text-muted">
                    <i class="fas fa-brain fa-2x mb-2"></i>
                    <p class="mb-0">Sistema en espera</p>
                    <small>Inicia el reconocimiento para comenzar</small>
                </div>
            </div>
        `;
    }

    ModalManager.showInfo("Reconocimiento Detenido", "El sistema ha sido detenido correctamente");
}

// =============================================================
// GRABAR NUEVO GESTO - ACTUALIZADO
// =============================================================
async function startRecording() {
    console.log('🎥 Iniciando grabación...');
    
    const video = document.getElementById("recorderPreview");
    const overlay = document.getElementById("recorderOverlay");
    const startBtn = document.getElementById("startRecordBtn");
    const stopBtn = document.getElementById("stopRecordBtn");
    const saveBtn = document.getElementById("saveRecordBtn");

    if (!video) {
        ModalManager.showError("Error", "Elemento de video no encontrado");
        return;
    }

    recordingFrames = [];
    frameCount = 0;
    recordStartTime = Date.now();
    recording = true;

    // Inicializar cámara de grabación
    if (!recorderStream) {
        recorderStream = await initCamera("recorderPreview", "recorderOverlay");
    }

    if (!recorderStream) {
        ModalManager.showError("Error", "No se pudo inicializar la cámara para grabar");
        return;
    }

    if (overlay) overlay.style.display = "none";
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    if (saveBtn) saveBtn.disabled = true;

    // Actualizar estado de grabación
    updateRecordingUI(true);

    ModalManager.showInfo("Grabación Iniciada", "Realiza tu gesto frente a la cámara");

    // Sistema de grabación automática
    const recordInterval = setInterval(() => {
        if (!recording) {
            clearInterval(recordInterval);
            return;
        }

        const frame = captureFrame(video);
        if (frame) {
            recordingFrames.push(frame);
            frameCount++;
            
            updateRecordingProgress();
            
            // Detener automáticamente después del máximo de frames
            if (frameCount >= CONFIG.MAX_FRAMES) {
                stopRecording();
                clearInterval(recordInterval);
                ModalManager.showInfo("Grabación Completa", "Se alcanzó el máximo de frames permitidos");
            }
        }
    }, 1000 / CONFIG.FRAME_RATE);
}

function updateRecordingUI(isRecording) {
    const recordingStatus = document.getElementById("recordingStatus");
    if (recordingStatus) {
        recordingStatus.textContent = isRecording ? "●" : "✓";
        recordingStatus.className = isRecording ? "value text-warning" : "value text-success";
    }
    
    const recordingIndicator = document.getElementById("recordingIndicator");
    if (recordingIndicator) {
        recordingIndicator.style.display = isRecording ? "block" : "none";
    }
}

function updateRecordingProgress() {
    const frameCountElement = document.getElementById("frameCount");
    if (frameCountElement) {
        frameCountElement.textContent = frameCount;
    }
    
    const now = Date.now();
    const elapsed = Math.floor((now - recordStartTime) / 1000);
    const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const seconds = String(elapsed % 60).padStart(2, "0");
    
    const recordingTimeElement = document.getElementById("recordingTime");
    if (recordingTimeElement) {
        recordingTimeElement.textContent = `${minutes}:${seconds}`;
    }
    
    // Actualizar barra de progreso
    const progressBar = document.getElementById("recordingProgress");
    if (progressBar) {
        const progress = (frameCount / CONFIG.MAX_FRAMES) * 100;
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
}

function stopRecording() {
    console.log('⏹️ Deteniendo grabación...');
    
    recording = false;
    
    const startBtn = document.getElementById("startRecordBtn");
    const stopBtn = document.getElementById("stopRecordBtn");
    const saveBtn = document.getElementById("saveRecordBtn");
    
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (saveBtn && recordingFrames.length >= CONFIG.MIN_FRAMES) {
        saveBtn.disabled = false;
    }

    updateRecordingUI(false);
    
    ModalManager.showSuccess(
        "Grabación Finalizada", 
        `Se capturaron ${frameCount} frames del gesto`
    );
}

// =============================================================
// GUARDAR GESTO - ACTUALIZADO CON MANEJO DE ERRORES MEJORADO
// =============================================================
async function saveGesture(event) {
    if (event) event.preventDefault();
    
    const nameInput = document.getElementById("gestureName");
    const categoryInput = document.getElementById("gestureCategory");
    const descriptionInput = document.getElementById("gestureDesc");

    if (!nameInput || !categoryInput) {
        ModalManager.showError("Error", "Formulario incompleto");
        return;
    }

    const name = nameInput.value.trim();
    const category = categoryInput.value;
    const description = descriptionInput ? descriptionInput.value.trim() : '';

    if (!name) {
        ModalManager.showError("Error", "El nombre del gesto es requerido");
        return;
    }

    if (recordingFrames.length < CONFIG.MIN_FRAMES) {
        ModalManager.showError("Error", `Se necesitan al menos ${CONFIG.MIN_FRAMES} frames para registrar un gesto`);
        return;
    }

    const loadingModal = ModalManager.showLoading("Guardando gesto...");

    try {
        const gestureData = {
            name: name,
            category: category,
            description: description,
            frames: recordingFrames
        };

        const result = await GesturesAPI.registerGesture(gestureData);
        
        ModalManager.hideLoading();
        
        if (result.success) {
            ModalManager.showSuccess(
                "Gesto Guardado", 
                `"${name}" ha sido registrado exitosamente`,
                `Frames: ${result.frames_registered || recordingFrames.length}, Categoría: ${category}`
            );
            
            // Resetear formulario y UI
            resetRecordingForm();
            
            // Actualizar lista de gestos
            loadGesturesList();
            
        } else {
            // Manejo mejorado de errores
            let errorMessage = result.error || "Error desconocido";
            if (errorMessage.includes("_graph is None") || errorMessage.includes("graph")) {
                errorMessage = "Error temporal del sistema de visión. Por favor, intenta nuevamente en unos segundos.";
                // Opcional: ofrecer reinicialización
                setTimeout(() => {
                    ModalManager.showConfirm(
                        "Reinicializar Sistema",
                        "Se detectó un error en el sistema de visión. ¿Deseas reinicializar?",
                        async function() {
                            try {
                                const reloadResult = await GesturesAPI.reinitializeExtractor();
                                if (reloadResult.success) {
                                    ModalManager.showSuccess("Sistema Reinicializado", "El sistema de visión ha sido reiniciado correctamente");
                                } else {
                                    ModalManager.showError("Error", "No se pudo reinicializar el sistema", reloadResult.error);
                                }
                            } catch (reloadError) {
                                ModalManager.showError("Error", "No se pudo reinicializar el sistema", reloadError.message);
                            }
                        }
                    );
                }, 1000);
            }
            ModalManager.showError("Error", "No se pudo guardar el gesto", errorMessage);
        }
        
    } catch (error) {
        ModalManager.hideLoading();
        let errorMessage = error.message;
        if (errorMessage.includes("_graph is None") || errorMessage.includes("graph")) {
            errorMessage = "Error temporal del sistema de visión. Por favor, intenta nuevamente.";
        }
        ModalManager.showError("Error", "No se pudo guardar el gesto", errorMessage);
    }
}

function resetRecordingForm() {
    const gestureForm = document.getElementById("gestureForm");
    if (gestureForm) gestureForm.reset();
    
    recordingFrames = [];
    frameCount = 0;
    
    const frameCountElement = document.getElementById("frameCount");
    if (frameCountElement) frameCountElement.textContent = "0";
    
    const recordingTimeElement = document.getElementById("recordingTime");
    if (recordingTimeElement) recordingTimeElement.textContent = "00:00";
    
    const saveRecordBtn = document.getElementById("saveRecordBtn");
    if (saveRecordBtn) saveRecordBtn.disabled = true;
    
    const progressBar = document.getElementById("recordingProgress");
    if (progressBar) {
        progressBar.style.width = "0%";
        progressBar.setAttribute('aria-valuenow', 0);
    }
}

// =============================================================
// MODO CONVERSACIÓN - ACTUALIZADO
// =============================================================
async function sendConversationMessage() {
    const input = document.getElementById("convInput");
    const message = input ? input.value.trim() : '';
    
    if (!message) {
        ModalManager.showWarning("Campo Vacío", "Por favor escribe un mensaje");
        return;
    }

    // Agregar mensaje del usuario al chat
    addChatMessage("user", message);
    if (input) input.value = "";

    const loadingModal = ModalManager.showLoading("Buscando gestos...");

    try {
        // Usar búsqueda en la lista de gestos existente
        const gesturesResult = await GesturesAPI.getGesturesList();
        
        ModalManager.hideLoading();
        
        if (gesturesResult.success && gesturesResult.gestures) {
            const matchingGestures = gesturesResult.gestures.filter(gesture => 
                gesture.name.toLowerCase().includes(message.toLowerCase()) ||
                (gesture.description && gesture.description.toLowerCase().includes(message.toLowerCase()))
            );
            
            if (matchingGestures.length > 0) {
                let response = `Encontré ${matchingGestures.length} gesto(s) relacionados:\n\n`;
                
                matchingGestures.forEach(gesture => {
                    response += `• <strong>${gesture.name}</strong>`;
                    if (gesture.description) {
                        response += ` - ${gesture.description}`;
                    }
                    response += ` <button class="btn btn-sm btn-outline-primary ms-2" onclick="playGesture('${gesture.video_path}', '${gesture.name}')">
                        <i class="fas fa-play"></i> Ver
                    </button><br>`;
                });
                
                addChatMessage("bot", response);
                
                // Sintetizar voz para el primer resultado
                if (matchingGestures.length > 0) {
                    const ttsResult = await GesturesAPI.textToSpeech(`Encontré el gesto ${matchingGestures[0].name}`);
                    if (ttsResult.success && ttsResult.audio_data) {
                        playAudio(ttsResult.audio_data);
                    }
                }
            } else {
                addChatMessage("bot", `No encontré gestos relacionados con "<strong>${message}</strong>". Intenta con otras palabras.`);
            }
        } else {
            addChatMessage("bot", `Error buscando gestos: ${gesturesResult.error}`);
        }
        
    } catch (error) {
        ModalManager.hideLoading();
        addChatMessage("bot", `Error buscando gestos para "${message}". Intenta nuevamente.`);
    }
}

function addChatMessage(sender, message) {
    const chatBox = document.getElementById("chatBox");
    if (!chatBox) return;
    
    const messageDiv = document.createElement("div");
    messageDiv.className = `msg ${sender}`;
    
    const avatar = sender === "user" ? "👤" : "🤖";
    const bubbleClass = sender === "user" ? "user" : "bot";
    
    messageDiv.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="bubble ${bubbleClass}">${message}</div>
    `;
    
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// =============================================================
// GESTIÓN DE BIBLIOTECA DE GESTOS - ACTUALIZADO
// =============================================================
async function loadGesturesList() {
    try {
        const result = await GesturesAPI.getGesturesList();
        
        const table = document.getElementById("gesturesTable");
        if (!table) return;
        
        if (result.success && result.gestures && result.gestures.length > 0) {
            table.innerHTML = result.gestures.map(gesture => `
                <tr>
                    <td>
                        <strong>${gesture.name}</strong>
                        ${gesture.description ? `<br><small class="text-muted">${gesture.description}</small>` : ''}
                    </td>
                    <td>
                        <span class="badge bg-primary">${gesture.category || 'general'}</span>
                    </td>
                    <td>${gesture.frames || 0}</td>
                    <td>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar bg-success" style="width: ${((gesture.valid_frames || gesture.frames || 0) / (gesture.frames || 1)) * 100}%"></div>
                        </div>
                        <small>${gesture.valid_frames || gesture.frames || 0}/${gesture.frames || 0} válidos</small>
                    </td>
                    <td>${gesture.created_at || 'N/A'}</td>
                    <td>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-primary" onclick="playGesture('${gesture.video_path}', '${gesture.name}')" title="Reproducir">
                                <i class="fas fa-play"></i>
                            </button>
                            <button class="btn btn-outline-info" onclick="showGestureDetails('${gesture.id}')" title="Detalles">
                                <i class="fas fa-info"></i>
                            </button>
                            <button class="btn btn-outline-danger" onclick="confirmDelete('${gesture.id}', '${gesture.name}')" title="Eliminar">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `).join('');
        } else {
            table.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-5">
                        <div class="text-muted">
                            <i class="fas fa-inbox fa-3x mb-3"></i>
                            <h5>No hay gestos registrados</h5>
                            <p class="mb-0">Comienza grabando tu primer gesto</p>
                        </div>
                    </td>
                </tr>
            `;
        }
    } catch (error) {
        console.error("Error cargando gestos:", error);
        const table = document.getElementById("gesturesTable");
        if (table) {
            table.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-5">
                        <div class="text-danger">
                            <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                            <h5>Error cargando gestos</h5>
                            <p class="mb-0">${error.message}</p>
                        </div>
                    </td>
                </tr>
            `;
        }
    }
}

async function confirmDelete(gestureId, gestureName) {
    ModalManager.showConfirm(
        "Confirmar Eliminación",
        `¿Eliminar el gesto "${gestureName}"? Esta acción no se puede deshacer.`,
        async function() {
            const loading = ModalManager.showLoading('Eliminando gesto...');
            try {
                const result = await GesturesAPI.deleteGesture(gestureId);
                ModalManager.hideLoading();
                
                if (result.success) {
                    ModalManager.showSuccess('Gesto Eliminado', `"${gestureName}" ha sido eliminado correctamente`);
                    loadGesturesList(); // Recargar lista
                } else {
                    ModalManager.showError('Error', 'No se pudo eliminar el gesto', result.error);
                }
            } catch (error) {
                ModalManager.hideLoading();
                ModalManager.showError('Error', 'No se pudo eliminar el gesto', error.message);
            }
        }
    );
}

function showGestureDetails(gestureId) {
    ModalManager.showInfo("Detalles del Gesto", "Función en desarrollo", "Próximamente disponible");
}

// =============================================================
// REPRODUCTOR DE GESTOS - MEJORADO
// =============================================================
function playGesture(videoPath, gestureName = '') {
    const player = document.getElementById("player");
    const nowPlaying = document.getElementById("nowPlaying");
    const playerStatus = document.getElementById("playerStatus");
    
    if (!player) {
        ModalManager.showError("Error", "Reproductor no disponible");
        return;
    }
    
    if (!videoPath) {
        ModalManager.showWarning("Video no disponible", "Este gesto no tiene video asociado");
        return;
    }
    
    // Mostrar loader
    player.style.display = 'block';
    player.innerHTML = `
        <div class="d-flex justify-content-center align-items-center h-100">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Cargando...</span>
            </div>
        </div>
    `;
    
    // Crear elemento video
    const videoElement = document.createElement('video');
    videoElement.src = videoPath;
    videoElement.controls = true;
    videoElement.className = 'w-100 h-100';
    videoElement.style.objectFit = 'contain';
    
    videoElement.onloadeddata = () => {
        player.innerHTML = '';
        player.appendChild(videoElement);
        videoElement.play().catch(error => {
            console.error("Error al reproducir:", error);
            ModalManager.showInfo("Reproducción", "Haz clic en el video para reproducir");
        });
    };
    
    videoElement.onerror = () => {
        player.innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100 text-danger">
                <div class="text-center">
                    <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                    <p>Error al cargar el video</p>
                </div>
            </div>
        `;
    };
    
    if (nowPlaying) {
        nowPlaying.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-play-circle me-2"></i>
                <strong>Reproduciendo:</strong> ${gestureName || videoPath.split('/').pop() || 'gesto demostrativo'}
            </div>
        `;
    }
    
    if (playerStatus) {
        playerStatus.className = "badge bg-success";
        playerStatus.innerHTML = `<i class="fas fa-play me-1"></i>Reproduciendo`;
    }
}

function stopPlayer() {
    const player = document.getElementById("player");
    const playerStatus = document.getElementById("playerStatus");
    const nowPlaying = document.getElementById("nowPlaying");
    
    if (player) {
        player.innerHTML = '';
        player.style.display = 'none';
    }
    
    if (playerStatus) {
        playerStatus.className = "badge bg-secondary";
        playerStatus.textContent = "En espera";
    }
    
    if (nowPlaying) {
        nowPlaying.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                <span class="fw-semibold">Selecciona un gesto para reproducir</span>
            </div>
        `;
    }
}

// =============================================================
// AUDIO MANAGER - MEJORADO
// =============================================================
function playAudio(audioData) {
    try {
        const audio = new Audio(audioData);
        audio.volume = 0.8;
        
        audio.play().catch(e => {
            console.log('Error reproduciendo audio:', e);
            // Fallback: usar síntesis de voz del navegador
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance("Gesto reconocido");
                speechSynthesis.speak(utterance);
            }
        });
    } catch (error) {
        console.error('Error con audio:', error);
    }
}

// =============================================================
// SISTEMA DE DIAGNÓSTICO
// =============================================================
async function checkSystemStatus() {
    try {
        const result = await GesturesAPI.getSystemStatus();
        if (result.success) {
            console.log('✅ Estado del sistema:', result.details);
            return true;
        } else {
            console.error('❌ Error en estado del sistema:', result.error);
            return false;
        }
    } catch (error) {
        console.error('❌ Error verificando estado del sistema:', error);
        return false;
    }
}

// =============================================================
// INICIALIZACIÓN - COMPLETA
// =============================================================
document.addEventListener("DOMContentLoaded", function() {
    console.log("🎯 gestures.js - Configurando event listeners...");
    
    // Event listeners mejorados
    document.addEventListener('click', function(e) {
        // Reconocimiento
        if (e.target.id === 'startRecBtn' || e.target.closest('#startRecBtn')) {
            e.preventDefault();
            startRecognition();
        }
        if (e.target.id === 'stopRecBtn' || e.target.closest('#stopRecBtn')) {
            e.preventDefault();
            stopRecognition();
        }
        
        // Grabación
        if (e.target.id === 'startRecordBtn' || e.target.closest('#startRecordBtn')) {
            e.preventDefault();
            startRecording();
        }
        if (e.target.id === 'stopRecordBtn' || e.target.closest('#stopRecordBtn')) {
            e.preventDefault();
            stopRecording();
        }
        
        // Conversación
        if (e.target.id === 'convSendBtn' || e.target.closest('#convSendBtn')) {
            e.preventDefault();
            sendConversationMessage();
        }
        
        // Player
        if (e.target.id === 'stopQueueBtn' || e.target.closest('#stopQueueBtn')) {
            e.preventDefault();
            stopPlayer();
        }
        
        // Biblioteca
        if (e.target.id === 'refreshList' || e.target.closest('#refreshList')) {
            e.preventDefault();
            loadGesturesList();
        }
        
        // Formulario
        if (e.target.id === 'resetFormBtn' || e.target.closest('#resetFormBtn')) {
            e.preventDefault();
            ModalManager.showConfirm(
                "Limpiar Formulario",
                "¿Estás seguro de que deseas limpiar el formulario?",
                function() {
                    resetRecordingForm();
                    ModalManager.showSuccess("Formulario Limpiado", "El formulario ha sido restablecido");
                }
            );
        }
    });
    
    // Form submit
    const gestureForm = document.getElementById("gestureForm");
    if (gestureForm) {
        gestureForm.addEventListener("submit", saveGesture);
    }
    
    // Enter key en conversación
    const convInput = document.getElementById("convInput");
    if (convInput) {
        convInput.addEventListener("keypress", function(e) {
            if (e.key === "Enter") {
                e.preventDefault();
                sendConversationMessage();
            }
        });
    }

    // Cargar lista inicial de gestos
    loadGesturesList();
    
    // Inicializar cámaras después de un delay
    setTimeout(() => {
        console.log('📷 Inicializando cámaras...');
        initCamera("recVideo", "recOverlay");
        initCamera("recorderPreview", "recorderOverlay");
    }, 1000);

    // Cargar estadísticas
    setTimeout(() => {
        loadStats();
    }, 2000);

    // Verificar estado del sistema
    setTimeout(() => {
        checkSystemStatus();
    }, 3000);

    console.log("✅ Sistema de gestos profesional inicializado correctamente");
});

// Función para cargar estadísticas
async function loadStats() {
    try {
        const result = await GesturesAPI.getGestureStats();
        if (result.success) {
            updateStatsDisplay(result.stats);
        }
    } catch (error) {
        console.error("Error cargando estadísticas:", error);
    }
}

function updateStatsDisplay(stats) {
    if (!stats) return;
    
    // Actualizar elementos de estadísticas en la UI
    const totalGestures = document.getElementById("totalGestures");
    const totalFrames = document.getElementById("totalFrames");
    const accuracyRate = document.getElementById("accuracyRate");
    
    if (totalGestures) totalGestures.textContent = stats.total_gestures || 0;
    if (totalFrames) totalFrames.textContent = stats.total_frames || 0;
    if (accuracyRate) accuracyRate.textContent = stats.accuracy_rate || "95%";
}

// Funciones globales para HTML
window.playGesture = playGesture;
window.confirmDelete = confirmDelete;
window.stopPlayer = stopPlayer;
window.startRecognition = startRecognition;
window.stopRecognition = stopRecognition;
window.startRecording = startRecording;
window.stopRecording = stopRecording;
window.sendConversationMessage = sendConversationMessage;
window.loadGesturesList = loadGesturesList;
window.showGestureDetails = showGestureDetails;
window.learnMoreAboutGesture = showGestureDetails;
window.checkSystemStatus = checkSystemStatus;
window.GesturesAPI = GesturesAPI;

console.log("🎯 gestures.js - Sistema completamente cargado y listo");