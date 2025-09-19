import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_heygen_api():
    api_key = os.getenv("HEYGEN_API_KEY")
    
    print(f"API Key: {api_key}")
    
    if not api_key or api_key == "your_heygen_api_key_here":
        print("ERROR: API key no configurada correctamente")
        return False
    
    # Payload actualizado para API v2
    payload = {
        "video_input": {
            "character": {
                "type": "avatar",
                "avatar_id": "11b6a3bad2e44e1896003b4a4f8b64c4"
            },
            "voice": {
                "type": "text",
                "input_text": "Hola, esto es una prueba de la API v2",
                "voice_id": "es_mx_001"
            },
            "background": {
                "type": "color",
                "color": "#FFFFFF"
            }
        },
        "dimension": {
            "width": 1280,
            "height": 720
        },
        "test": True,
        "watermark": {
            "enable": False
        }
    }
    
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # URL correcta para API v2
    api_url = "https://api.heygen.com/v2/video/generate"
    
    try:
        print("Enviando solicitud a HeyGen v2 API...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API v2 funciona correctamente")
            print(f"Video URL: {data.get('data', {}).get('video_url', 'No disponible')}")
            return True
        else:
            print("❌ Error en la API v2")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Respuesta: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

def test_heygen_status():
    """Probar el endpoint de status de HeyGen"""
    api_key = os.getenv("HEYGEN_API_KEY")
    
    if not api_key:
        print("No API key configured")
        return False
    
    headers = {"X-Api-Key": api_key}
    
    try:
        print("Probando endpoint de status...")
        response = requests.get("https://api.heygen.com/v1/status", headers=headers, timeout=10)
        print(f"Status API Response: {response.status_code}")
        print(f"Status Content: {response.text[:200]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error en status check: {e}")
        return False

if __name__ == "__main__":
    print("=== Test de HeyGen API v2 ===")
    test_heygen_api()
    
    print("\n=== Test de Status ===")
    test_heygen_status()