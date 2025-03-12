import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


#def cargar_documentos(ruta_archivo):
#    if not os.path.exists(ruta_archivo):
#        raise FileNotFoundError(f"El archivo {ruta_archivo} no existe")
#    
#    carga = PyPDFLoader(ruta_archivo)
#    documentos = carga.load()
#    texto_spliter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap = 500)
#    documentos = texto_spliter.split_documents(texto_spliter)
    
#    return documentos

def cargar_documentos(rutas_archivos):
    """
    Carga y divide múltiples documentos PDF en fragmentos de texto.
    Soporta tanto una lista de rutas de archivos como una carpeta con múltiples PDFs.

    Args:
        rutas_archivos (str o list): Ruta a un archivo PDF, lista de rutas de archivos PDF,
                                     o ruta a una carpeta que contiene PDFs.

    Returns:
        list: Lista de documentos divididos en fragmentos.
    """
    # Lista para almacenar todos los documentos cargados y divididos
    todos_los_documentos = []

    # Si se proporciona una sola ruta (str), convertirla en una lista
    if isinstance(rutas_archivos, str):
        rutas_archivos = [rutas_archivos]

    # Recorrer cada ruta de archivo o carpeta
    for ruta in rutas_archivos:
        # Si es una carpeta, obtener todos los archivos PDF dentro de ella
        if os.path.isdir(ruta):
            print(f"Procesando carpeta: {ruta}")
            archivos_en_carpeta = [
                os.path.join(ruta, archivo) for archivo in os.listdir(ruta)
                if archivo.endswith(".pdf")
            ]
            if not archivos_en_carpeta:
                print(f"Advertencia: No se encontraron archivos PDF en la carpeta {ruta}.")
            rutas_archivos.extend(archivos_en_carpeta)  # Agregar los archivos a la lista
            continue  # Saltar al siguiente elemento en la lista

        # Si es un archivo PDF, cargarlo
        if ruta.endswith(".pdf"):
            # Validar que el archivo exista
            if not os.path.exists(ruta):
                print(f"Advertencia: El archivo {ruta} no existe.")
                continue

            # Cargar el PDF
            print(f"Procesando archivo: {ruta}")
            ##################################################
            #carga = PyPDFLoader(ruta)
            #documentos = carga.load()
            ## O
            try:
                carga = PyPDFLoader(ruta)
                documentos = carga.load()
            except Exception as e:
                print(f"Error al cargar el archivo {ruta}: {e}")
                continue
            #################################################
            
            # Validar que se hayan cargado documentos
            if not documentos:
                print(f"Advertencia: No se pudieron cargar documentos desde el archivo {ruta}.")
                continue

            # Dividir los documentos en fragmentos
            ####################################################################
            #texto_spliter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            #documentos_divididos = texto_spliter.split_documents(documentos)
            
            ## O
            
            # Dividir los documentos en fragmentos
            try:
                texto_spliter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
                documentos_divididos = texto_spliter.split_documents(documentos)
            except Exception as e:
                print(f"Error al dividir los documentos del archivo {ruta}: {e}")
                continue
            ####################################################################
            
            # Validar que los fragmentos tengan el tamaño correcto
            #for doc in documentos_divididos:
            #    if len(doc.page_content) < 100:  # Ajusta este valor según tu caso
            #        print(f"Advertencia: Fragmento muy pequeño encontrado en {ruta}: {doc.page_content}")
            
            # Agregar los documentos divididos a la lista general
            todos_los_documentos.extend(documentos_divididos)
            print(f"Se cargaron {len(documentos_divididos)} fragmentos desde {ruta}.")
        else:
            print(f"Advertencia: La ruta {ruta} no es un archivo PDF.")
    
    # Validar que se hayan cargado documentos en total
    if not todos_los_documentos:
        raise ValueError("No se pudieron cargar documentos desde ninguno de los archivos o carpetas proporcionados.")
    
    print(f"Se cargaron un total de {len(todos_los_documentos)} fragmentos de texto.")
    
    return todos_los_documentos


#ruta_files = [
#    "pdf_solos/ASIS_REPORTE_EJECUTIVO.pdf",
#    "pdf_carpetas/pdfs_varios"
#]

#documentos = cargar_documentos(ruta_files)

def crear_vectorstore(docs):
    """
    Crea un vector store a partir de una lista de documentos.

    Args:
        docs (list): Lista de documentos divididos en fragmentos.

    Returns:
        Chroma: Vector store creado.
    """
    # Validar que se proporcionen documentos
    if not docs:
        raise ValueError("No se proporcionaron documentos para crear el vector store.")

    # Crear el modelo de embeddings
    try:
        # Busqueda semantica, basada en similitud de embedings
        # Modelo compatible con FastEmbedEmbeddings
        #embeding_modelo = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Modelo de Hugging Face
        embeding_modelo = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error al crear el modelo de embeddings: {e}")
        raise

    # Crear el vector store
    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeding_modelo,
            persist_directory="1chroma_ms4m_bd_vfche",
            collection_name="1data_ms4m_vfche" #colection dentro del vs
        )
        print("Vector store creado correctamente.")
    except Exception as e:
        print(f"Error al crear el vector store: {e}")
        raise

    # Depuración: Verificar el número de documentos en el vector store
    print(f"Número de documentos en el vector store: {vector_store._collection.count()}")

    # Validar que el vector store no esté vacío
    if vector_store._collection.count() == 0:
        raise ValueError("El vector store está vacío. No se cargaron documentos.")

    # Realizar una búsqueda de prueba
    try:
        query = "Area de planeamiento de ms4m"  # Cambia esto por una consulta relevante para tu caso
        resultados = vector_store.similarity_search(query, k=3)
        print(f"Búsqueda de prueba realizada. Resultados encontrados: {len(resultados)}")
    except Exception as e:
        print(f"Error al realizar la búsqueda de prueba: {e}")
        raise

    return vector_store



# Cargar documentos
#ruta_files = [
#    "pdf_solos/ASIS_REPORTE_EJECUTIVO.pdf",
#    "pdf_carpetas/pdfs_varios"
#]
#documentos = cargar_documentos(ruta_files)

# Crear el vector store
#vector_store = crear_vectorstore(documentos)
    