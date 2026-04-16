from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from datasets import load_dataset

# Carga variables de entorno (PG_CONN_STR, etc.)
load_dotenv()
EMBEDDING_MODEL_NAME = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"


# FUENTE DE DATOS → DOCUMENT

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
documents = [
    Document(content=doc["content"], meta=doc["meta"])
    for doc in dataset
]


# COMPONENTES

# Embedder: transforma texto → embedding vectorial
document_embedder = OllamaDocumentEmbedder(
    model=EMBEDDING_MODEL_NAME,
    url=OLLAMA_BASE_URL,
    batch_size=32,
)

# Document Store: persistencia en PostgreSQL + pgvector
vector_document_store = PgvectorDocumentStore(
    table_name="haystack_docs_prueba",
    embedding_dimension=768,  # debería derivarse dinámicamente en producción
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)

# Writer: adapta Document → formato persistible en el store
writer = DocumentWriter(document_store=vector_document_store)


# PIPELINE DE INDEXACIÓN

pipeline = Pipeline()
pipeline.add_component("embedder", document_embedder)
pipeline.add_component("writer", writer)
pipeline.connect("embedder.documents", "writer.documents")


pipeline.run({
    "embedder": {
        "documents": documents
    }
})