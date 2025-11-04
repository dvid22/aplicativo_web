# 🧏‍♂️ Sistema de Traducción del Lenguaje de Señas como Herramienta para la Educación Inclusiva

Este proyecto tiene como objetivo promover la **inclusión educativa** mediante el desarrollo de una **aplicación web inteligente** capaz de **traducir el lenguaje de señas colombiano (LSC)** a texto y voz en tiempo real.
La herramienta está orientada a facilitar la comunicación entre personas sordas y oyentes dentro de contextos educativos y sociales.

---

## 🚀 Características principales

* ✋ **Reconocimiento de señas en tiempo real** utilizando **MediaPipe** y **OpenCV**.
* 🧠 **Procesamiento inteligente** para traducir los movimientos de las manos a texto.
* 🔊 **Conversión a voz** mediante síntesis de audio.
* 👥 **Autenticación de usuarios** con inicio de sesión tradicional y con **Google OAuth**.
* 💾 **Base de datos en la nube (MongoDB Atlas)** para almacenar registros e historial de traducciones.
* 🌐 **Interfaz web moderna y adaptable**, desarrollada con **Flask**, **HTML5**, **CSS3** y **JavaScript**.
* 🔐 **Sistema de login y registro** seguro, con manejo de sesiones y roles de usuario.

---

## 🧩 Arquitectura del sistema

El proyecto está desarrollado bajo el patrón **Modelo–Vista–Controlador (MVC)** e integra los siguientes componentes:

* **Frontend:** HTML, CSS, Bootstrap, JavaScript.
* **Backend:** Python (Flask Framework).
* **Base de datos:** MongoDB Atlas.
* **IA y Visión por Computador:** MediaPipe, OpenCV.
* **Autenticación:** Flask-Login, Google OAuth 2.0.

---

## 🛠️ Tecnologías utilizadas

| Categoría                  | Tecnologías                        |
| -------------------------- | ---------------------------------- |
| **Lenguaje principal**     | Python 3.11                        |
| **Framework backend**      | Flask                              |
| **Frontend**               | HTML5, CSS3, JavaScript, Bootstrap |
| **Base de datos**          | MongoDB Atlas                      |
| **IA / Visión artificial** | MediaPipe, OpenCV                  |
| **Autenticación**          | Flask-Login, OAuth 2.0             |
| **Despliegue**             | Render / Localhost                 |

---

## ⚙️ Instalación y configuración

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/usuario/nombre-del-repositorio.git
cd nombre-del-repositorio
```

### 2️⃣ Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate   # En Linux / Mac
venv\Scripts\activate      # En Windows
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4️⃣ Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
SECRET_KEY=tu_clave_segura
MONGO_URI=tu_conexion_mongodb
GOOGLE_CLIENT_ID=tu_id_cliente_google
GOOGLE_CLIENT_SECRET=tu_secreto_google
```

### 5️⃣ Ejecutar la aplicación

```bash
flask run
```

Luego abre en tu navegador:
👉 [http://localhost:5000](http://localhost:5000)

---

## 💡 Uso del sistema

1. Inicia sesión con correo o cuenta de Google.
2. Accede al módulo de traducción.
3. Activa la cámara para que el sistema detecte los movimientos de tus manos.
4. Observa la traducción en texto y escucha la conversión a voz.
5. Consulta tu historial o cierra sesión cuando termines.


---

## 🌍 Despliegue en la nube

El proyecto puede ser desplegado fácilmente en plataformas como **Render**, **Railway** o **Heroku**, configurando las variables de entorno correspondientes y enlazando el repositorio desde GitHub.

---

## 🧠 Propósito educativo

El sistema contribuye a la **inclusión social y educativa** de las personas con discapacidad auditiva, permitiendo una interacción más equitativa en entornos académicos y tecnológicos.
Busca además servir como base para futuras investigaciones en **inteligencia artificial aplicada a la accesibilidad**.

---

## 📜 Licencia

Este proyecto se distribuye bajo la licencia **MIT License**, lo que permite su uso y modificación con fines académicos y de investigación.

---

## 📸 Vista previa

![Demo del sistema](https://via.placeholder.com/800x400.png?text=Vista+Previo+del+Sistema)

> *“La inclusión comienza cuando la tecnología se pone al servicio de todos.”*
