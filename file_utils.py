import pdfplumber
from docx import Document

def leer_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text

def leer_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def leer_pdf(file_path):
    textos = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 0:
                textos.append(text)
    return textos

def fragmentar_texto(texto, tamaño=800, solapamiento=100):
    fragmentos = []
    inicio = 0
    longitud = len(texto)
    while inicio < longitud:
        fin = min(inicio + tamaño, longitud)
        fragmentos.append(texto[inicio:fin])
        if fin == longitud:
            break
        inicio += (tamaño - solapamiento)
    return fragmentos
