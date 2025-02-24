import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat2_llm import iniciar_llm_chat  # Importa tu función principal

# Inicializa la aplicación FastAPI
app = FastAPI()

# Define el modelo de entrada para la API
class QueryRequest(BaseModel):
    question: str

# Inicializa el modelo RAG (esto se ejecuta una vez al iniciar la API)
ruta_files = [
    "pdf_solos/ASIS_REPORTE_EJECUTIVO.pdf",
    "pdf_carpetas/pdfs_varios"
]
cadena_rag = iniciar_llm_chat(ruta_files)

# Define el endpoint para recibir preguntas
@app.post("/respuesta_llm")
def ask_question(query: QueryRequest):
    try:
        # Obtén la respuesta del modelo RAG
        respuesta = cadena_rag.invoke({"query": query.question})
        return {
            "response": respuesta["result"],
            "source_documents": [
                {
                    "page": doc.metadata["page"],
                    "source": doc.metadata["source"]
                }
                for doc in respuesta["source_documents"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecuta la API con Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
