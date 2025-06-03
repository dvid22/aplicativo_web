// static/js/webcam.js

const video = document.getElementById('video');

// Solicitar acceso a la cámara
navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
  })
  .catch((err) => {
    console.error("No se pudo acceder a la cámara:", err);
    alert("Error: No se pudo acceder a la cámara.");
  });

// Reconocimiento de gesto (mock)
function recognizeGesture() {
  document.getElementById('recognizedText').textContent = "Gesto detectado: 👍"; // Ejemplo
  // Aquí puedes integrar MediaPipe o enviar una captura al backend para analizar
}

// Registro de gesto (mock)
function registerGesture() {
  const name = document.getElementById('gestureName').value.trim();
  if (!name) {
    alert("Por favor, ingresa un nombre para el gesto.");
    return;
  }

  // Aquí podrías enviar los datos al backend (POST) para guardarlos en MongoDB
  alert(`Gesto "${name}" registrado correctamente (simulado).`);
  document.getElementById('gestureName').value = '';
}
