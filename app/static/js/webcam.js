// static/js/webcam.js

// Variables globales para manejar el stream de la cámara
let cameraStream = null;
const video = document.getElementById('video');

// Función para activar la cámara
export async function activateCamera() {
  try {
    // Verificar si ya hay un stream activo
    if (cameraStream) {
      console.log("La cámara ya está activada");
      return;
    }

    // Solicitar acceso a la cámara
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        facingMode: 'user', // Preferir cámara frontal
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    
    video.srcObject = stream;
    cameraStream = stream;
    
    // Esperar a que el video esté listo para reproducir
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.play();
        resolve(true);
      };
    });
    
  } catch (err) {
    console.error("Error al acceder a la cámara:", err);
    alert("No se pudo acceder a la cámara. Asegúrate de haber dado los permisos necesarios.");
    return false;
  }
}

// Función para desactivar la cámara
export function deactivateCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    cameraStream = null;
  }
}

// Función para capturar un frame del video
export function captureFrame() {
  if (!cameraStream) {
    alert("Primero debes activar la cámara");
    return null;
  }
  
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  return canvas.toDataURL('image/jpeg');
}

// Reconocimiento de gesto (mejorado)
export async function recognizeGesture() {
  try {
    if (!cameraStream) {
      await activateCamera();
    }
    
    const imageData = captureFrame();
    if (!imageData) return;
    
    // Aquí puedes enviar imageData al backend para análisis
    // Ejemplo simulado:
    document.getElementById('recognizedText').textContent = "Analizando gesto...";
    
    // Simular demora de análisis
    setTimeout(() => {
      document.getElementById('recognizedText').textContent = "Gesto detectado: 👍";
    }, 1500);
    
  } catch (err) {
    console.error("Error en reconocimiento de gesto:", err);
    alert("Error al reconocer el gesto");
  }
}

// Registro de gesto (mejorado)
export async function registerGesture() {
  try {
    const name = document.getElementById('gestureName').value.trim();
    if (!name) {
      alert("Por favor, ingresa un nombre para el gesto.");
      return;
    }

    const imageData = captureFrame();
    if (!imageData) return;

    // Aquí podrías enviar {name, imageData} al backend para guardar en MongoDB
    console.log(`Registrando gesto "${name}" con imagen:`, imageData.substring(0, 30) + "...");
    
    // Simular registro exitoso
    alert(`Gesto "${name}" registrado correctamente!`);
    document.getElementById('gestureName').value = '';
    
  } catch (err) {
    console.error("Error al registrar gesto:", err);
    alert("Error al registrar el gesto");
  }
}

// Limpieza al salir
window.addEventListener('beforeunload', deactivateCamera);