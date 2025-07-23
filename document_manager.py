import os
from file_utils import leer_docx, leer_txt, leer_pdf, fragmentar_texto

class DocumentManager:
    def __init__(self, docs_folder):
        self.docs_folder = docs_folder

    def cargar_fragmentos_archivo(self, file_path, file_type, file_name, tamaño, solapamiento):
        fragmentos = []
        metadatos = []
        try:
            if file_type == 'pdf':
                paginas = leer_pdf(file_path)
                for i, text in enumerate(paginas):
                    if text and len(text) > 100:
                        frags = fragmentar_texto(text, tamaño, solapamiento)
                        for j, frag in enumerate(frags):
                            fragmentos.append(frag)
                            metadatos.append({
                                "file": file_name,
                                "page": i+1,
                                "fragment": j+1
                            })
            elif file_type == 'docx':
                text = leer_docx(file_path)
                if text and len(text) > 100:
                    frags = fragmentar_texto(text, tamaño, solapamiento)
                    for j, frag in enumerate(frags):
                        fragmentos.append(frag)
                        metadatos.append({
                            "file": file_name,
                            "page": "-",
                            "fragment": j+1
                        })
            elif file_type == 'txt':
                text = leer_txt(file_path)
                if text and len(text) > 100:
                    frags = fragmentar_texto(text, tamaño, solapamiento)
                    for j, frag in enumerate(frags):
                        fragmentos.append(frag)
                        metadatos.append({
                            "file": file_name,
                            "page": "-",
                            "fragment": j+1
                        })
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")
        return fragmentos, metadatos

    def cargar_fragmentos(self, tamaño, solapamiento):
        fragmentos = []
        metadatos = []
        for root, _, files in os.walk(self.docs_folder):
            for file in files:
                file_path = os.path.join(root, file)
                ext = file.lower().split('.')[-1]
                if ext in ['pdf', 'docx', 'txt']:
                    frags, metas = self.cargar_fragmentos_archivo(file_path, ext, file, tamaño, solapamiento)
                    fragmentos.extend(frags)
                    metadatos.extend(metas)
        return fragmentos, metadatos
