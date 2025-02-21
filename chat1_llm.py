from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from cargar_data import cargar_documentos, crear_vectorstore
from langchain_community.vectorstores import Chroma
#from langchain_community.vectorstores import PGVector

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# Codigo para colores ANSI
AZUL = "\033[94m"
VERDE = "\033[92m" 
RESET = "\033[0m"


def iniciar_llm_chat(ruta_files):
    llm_ms4m = Ollama(model="mistral")
    # Modelo de Hugging Face
    embeding_modelo = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Intentar cargar el vector store existente
    try:
        vs_ms4m = Chroma(embedding_function=embeding_modelo,
                         persist_directory="chroma_ms4m_basedatos",
                 collection_name="data_ms4m_despacho")
    except Exception as e:
        print(f"No se pudo cargar el vector store existente: {e}")
        print("Creando un nuevo vector store...")
        # Cargar documentos y crear el vector store
        documentos = cargar_documentos(ruta_files)
        vs_ms4m = crear_vectorstore(documentos)
    # Crear el retriever, para que la busqueda se use el vector store que hemos creado
    retriver_ms4m = vs_ms4m.as_retriever(search_kwargs={"k":3})
    
    
    prompt_template_ms4m = """Usa la siguiente informacion para responder al usuario.
    Si no sabes algo, no inventes una respuesta solo en esos casos menciona que no lo sabes.

    Contexto : {context}
    Pregunta : {question}

    Solo devuelve la respuesta util y responde unicamente en el idioma español
    Respuesta Util:
    """  
    
    # Le decimos cual es el prompt, y que deberia usar como inputs el contexto y la pregunta
    prompt_1 = PromptTemplate(template=prompt_template_ms4m,
                            input_variables=["context", "question"])
    
    
    # crear una cadena de RetrievalQA en LangChain 
    cadena_rag = RetrievalQA.from_chain_type(
    llm=llm_ms4m,                      # Modelo de lenguaje (por ejemplo, Mistral)
    chain_type="stuff",                # Tipo de cadena (en este caso, "stuff")
    retriever=retriver_ms4m,           # Retriever que busca los documentos relevantes
    return_source_documents=True,      # Devuelve los documentos fuente utilizados
    chain_type_kwargs={"prompt": prompt_1}  # Prompt personalizado
    )
    
    print("Bienvenido al Copilot_MS4M, Escribe 'salir' para terminar la sesion")
    
    while True:
        pregunta = input(f"{AZUL}Tú:{RESET}")
        if pregunta.lower() == "salir":
            print("¡Nos vemos luego")
            break
        
        # 9. Obtener la respuesta del modelo
        try:
            respuesta = cadena_rag.invoke({"query": pregunta})
            print(f"{VERDE}Asistente Copilot:{RESET}", respuesta["result"])

            # 10. Mostrar los documentos fuente utilizados
            if respuesta["source_documents"]:
                print("Documentos fuente utilizados:")
                for doc in respuesta["source_documents"]:
                    print(f"- Página {doc.metadata['page']} de {doc.metadata['source']}")
            else:
                print("No se encontraron documentos relevantes.")
        except Exception as e:
            print(f"Error al generar la respuesta: {e}")

        #########################################################################
        #respuesta = cadena_rag.invoke({"query": pregunta})
        
        #metadata = []

        #for _ in respuesta["source_documents"]:
        #    metadata.append((_.metadata["page"], _.metadata["source"]))
        #print(f"{VERDE}Asistente Copilot:{RESET}", respuesta["result"], "\n", metadata)
        ##############################################################################

if __name__ == "__main__":
    ruta_files2 = [
    "pdf_solos/ASIS_REPORTE_EJECUTIVO.pdf",
    "pdf_carpetas/pdfs_varios"
    ]
    iniciar_llm_chat(ruta_files2) 
    
    
