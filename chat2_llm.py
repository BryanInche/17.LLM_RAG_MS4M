from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from cargar_solodata import cargar_documentos, crear_vectorstore
#from langchain_community.vectorstores import PGVector

def iniciar_llm_chat(ruta_files):
    # Inicializar el modelo de lenguaje (Mistral)
    llm_ms4m = Ollama(model="mistral")
    
    # Inicializar el modelo de embeddings (Hugging Face)
    embeding_modelo = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Intentar cargar el vector store existente
    try:
        vs_ms4m = Chroma(
            embedding_function=embeding_modelo,
            persist_directory="1chroma_ms4m_bd_vfche",
            collection_name="1data_ms4m_vfche"
        )
    except Exception as e:
        print(f"No se pudo cargar el vector store existente: {e}")
        print("Creando un nuevo vector store...")
        # Cargar documentos y crear el vector store
        documentos = cargar_documentos(ruta_files)
        vs_ms4m = crear_vectorstore(documentos)
    
    # Crear el retriever para buscar en el vector store
    retriver_ms4m = vs_ms4m.as_retriever(search_kwargs={"k": 3})
    
    # Definir el prompt personalizado
    prompt_template_ms4m = """
    Usa la siguiente información para responder al usuario.
    Si no sabes algo, no inventes una respuesta, solo menciona que no lo sabes.

    Contexto: {context}
    Pregunta: {question}

    Solo devuelve la respuesta útil y responde únicamente en el idioma español.
    Respuesta útil:
    """
    
    prompt_1 = PromptTemplate(
        template=prompt_template_ms4m,
        input_variables=["context", "question"]
    )
    
    # Crear la cadena de RetrievalQA
    cadena_rag = RetrievalQA.from_chain_type(
        llm=llm_ms4m,                      # Modelo de lenguaje (Mistral)
        chain_type="stuff",                # Tipo de cadena ("stuff")
        retriever=retriver_ms4m,           # Retriever para buscar documentos
        return_source_documents=True,      # Devuelve los documentos fuente
        chain_type_kwargs={"prompt": prompt_1}  # Prompt personalizado
    )
    
    # Devolver la cadena de RetrievalQA para su uso en la API
    return cadena_rag

