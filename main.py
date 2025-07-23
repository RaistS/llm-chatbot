import threading

import gradio as gr
from document_manager import DocumentManager
from embedding_manager import EmbeddingManager
from chromadb_manager import ChromaDBManager
from ollama_utils import llamar_ollama_stream, limpiar_pensamientos

DOCS_FOLDER = r"F:\LLM Urgencias\docs"
EMBED_CACHE_DIR = "embed_cache"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "deepseek-r1"
BATCH_SIZE = 32

# Inicialización de módulos
document_manager = DocumentManager(DOCS_FOLDER)
embedding_manager = EmbeddingManager(EMBEDDING_MODEL, EMBED_CACHE_DIR)
chroma_manager = ChromaDBManager(CHROMA_DB_PATH, "docs_llm")

# --- Gradio UI & lógica de chat ---
def procesar_archivos(files, chunk, overlap):
    nuevos = 0
    for file in files:
        file_path = file.name
        file_name = file.name.split("\\")[-1].split("/")[-1]
        ext = file_name.lower().split('.')[-1]
        if ext not in ['pdf', 'docx', 'txt']:
            continue
        # Guardar el archivo en la carpeta DOCS_FOLDER
        destino = f"{DOCS_FOLDER}/{file_name}"
        with open(destino, "wb") as f_out, open(file_path, "rb") as f_in:
            f_out.write(f_in.read())
        frags, metas = document_manager.cargar_fragmentos_archivo(destino, ext, file_name, chunk, overlap)
        existing_ids = set(chroma_manager.get_existing_ids())
        nuevos += chroma_manager.indexar_batch(frags, metas, embedding_manager, existing_ids)
    return f"Archivos procesados e indexados. Fragmentos nuevos: {nuevos}"

def reindexar_todo(tamaño, solapamiento, output):
    def reindex_bg():
        fragmentos, metadatos = document_manager.cargar_fragmentos(tamaño, solapamiento)
        existing_ids = set(chroma_manager.get_all_ids())
        nuevos = chroma_manager.indexar_batch(fragmentos, metadatos, embedding_manager, existing_ids)

        print(f"Reindexado completo. Fragmentos nuevos: {nuevos}")
    threading.Thread(target=reindex_bg).start()
    # Devuelve un mensaje rápido para el usuario
    return "Reindexado lanzado en background. Puedes seguir usando el chat."


def buscar_fragmentos_semanticos(pregunta, n):
    pregunta_emb = embedding_manager.embed(pregunta)
    resultados = chroma_manager.collection.query(
        query_embeddings=[pregunta_emb],
        n_results=n
    )
    docs = resultados["documents"][0]
    metadatas_res = resultados["metadatas"][0]
    return docs, metadatas_res

def chat_llm_gradio(pregunta, historial, n_frags, max_frag_chars):
    # --- Recuperar contexto (semántico puro por simplicidad) ---
    fragmentos, metadatas = buscar_fragmentos_semanticos(pregunta, n=n_frags)
    contexto = ""
    citas = []
    for idx, (frag, meta) in enumerate(zip(fragmentos, metadatas), 1):
        recortado = frag[:max_frag_chars] + "..." if len(frag) > max_frag_chars else frag
        contexto += f"\n[{idx}] {recortado}"
        ref = f"{meta['file']} (página {meta['page']}, fragmento {meta['fragment']})"
        citas.append(f"[{idx}] {ref}")

    prompt = (
        "Responde en español, SOLO usando los fragmentos citados. "
        "Cuando uses información de un fragmento, indica la cita [1], [2], etc. "
        "No expliques tu razonamiento. No incluyas texto entre <think>.\n"
        f"Fragmentos:\n{contexto}\n\nPregunta: {pregunta}\n"
    )
    respuesta = ""
    for fragment in llamar_ollama_stream(prompt, modelo=MODEL_NAME):
        respuesta += fragment
        yield respuesta, None, historial
    respuesta = limpiar_pensamientos(respuesta)
    respuesta += "\n\nReferencias utilizadas:\n" + "\n".join(citas)
    historial.append({"pregunta": pregunta, "respuesta": respuesta})
    historial_vis = ""
    for h in historial:
        historial_vis += f"\n**Usuario:** {h['pregunta']}\n**AI:** {h['respuesta']}\n"
    yield respuesta, historial_vis, historial

with gr.Blocks() as iface:
    gr.Markdown("# Chat semántico sobre tus documentos\n\nSube tus PDFs, Word o TXT y luego pregunta lo que quieras.")

    with gr.Row():
        uploader = gr.File(file_count="multiple", file_types=[".pdf", ".docx", ".txt"], label="Sube tus documentos aquí")
        upload_btn = gr.Button("Procesar e indexar documentos")
        reindex_btn = gr.Button("Reindexar toda la carpeta")
    output_upload = gr.Textbox(label="Resultado de carga", interactive=False)

    chunk_slider = gr.Slider(400, 2000, value=800, step=50, label="Tamaño de fragmento")
    overlap_slider = gr.Slider(0, 400, value=100, step=10, label="Solapamiento de fragmentos")
    nfrags_slider = gr.Slider(1, 5, value=2, step=1, label="Fragmentos para la respuesta")
    fragchars_slider = gr.Slider(100, 2000, value=600, step=50, label="Máx. longitud de fragmento (respuesta)")

    upload_btn.click(
        lambda files, chunk, overlap: procesar_archivos(files, int(chunk), int(overlap)),
        inputs=[uploader, chunk_slider, overlap_slider],
        outputs=output_upload
    )
    reindex_btn.click(
        lambda chunk, overlap: reindexar_todo(int(chunk), int(overlap), output_upload),
        inputs=[chunk_slider, overlap_slider],
        outputs=output_upload
    )

    gr.Markdown("---")

    pregunta = gr.Textbox(lines=3, placeholder="Haz tu pregunta sobre los documentos...")
    respuesta = gr.Textbox(label="Respuesta", interactive=False)
    historial_vis = gr.Markdown(label="Historial de chat")
    state_hist = gr.State([])

    preguntar_btn = gr.Button("Preguntar")
    preguntar_btn.click(
        chat_llm_gradio,
        inputs=[pregunta, state_hist, nfrags_slider, fragchars_slider],
        outputs=[respuesta, historial_vis, state_hist]
    )

if __name__ == "__main__":
    iface.launch()
