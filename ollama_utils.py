import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"

def limpiar_pensamientos(texto):
    """Limpia etiquetas <think> y caracteres especiales del output de Ollama."""
    texto = re.sub(r'(<|\\u003c)think(>|\\u003e).*?(<|\\u003c)/think(>|\\u003e)', '', texto, flags=re.DOTALL | re.IGNORECASE)
    texto = texto.replace('\\n', '\n').replace('\\u003c', '<').replace('\\u003e', '>')
    return texto.strip()

def llamar_ollama_stream(prompt, modelo="deepseek-r1"):
    """
    Consulta a Ollama en modo streaming, retorna fragmentos a medida que llegan.
    Devuelve un generador que yield cada fragmento recibido.
    """
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": True
    }
    respuesta = ""
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    try:
                        part = re.search(r'"response"\s*:\s*"([^"]*)"', data)
                        if part:
                            fragment = part.group(1)
                            respuesta += fragment
                            yield fragment
                    except Exception:
                        continue
    except Exception as err:
        yield f"Error conectando al modelo: {err}"

def llamar_ollama(prompt, modelo="deepseek-r1"):
    """
    Consulta simple a Ollama (sin streaming).
    """
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        data = r.json()
        respuesta = data.get("response", "")
        return limpiar_pensamientos(respuesta)
    except Exception as err:
        return f"Error conectando al modelo: {err}"
