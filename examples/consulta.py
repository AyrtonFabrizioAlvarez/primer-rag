from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from dotenv import load_dotenv

# Carga variables de entorno (PG_CONN_STR, etc.)
load_dotenv()
EMBEDDING_MODEL_NAME = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME1="mistral"
MODEL_NAME2="phi3"
MODEL_NAME3="qwen3.5"

# ME CONECTO A LA BD VECTORIAL
vector_document_store = PgvectorDocumentStore(
    table_name="haystack_docs_prueba",
    embedding_dimension=768,
    vector_function="cosine_similarity",
    search_strategy="hnsw",
)

# INSTANCIO EL EMBEDDER PARA LA PREGUNTA
text_embedder = OllamaTextEmbedder(
    model=EMBEDDING_MODEL_NAME,
    url=OLLAMA_BASE_URL
)

# INSTANCIO EL RETRIEVER PARA IR A BUSCAR LOS SIMILARES A LA BD VECTORIAL
retriever = PgvectorEmbeddingRetriever(document_store=vector_document_store, top_k=2)

# ESTABLEZCO EL TEMPLATE DE RESPUESTA
template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
"""

# INSTANCIO COMO VA A SER EL ARMADO DEL PROMT CON EL TEMPLATE QUE DEFINI ANTES
prompt_builder = PromptBuilder(
    template=template,
    required_variables=["documents", "question"]
)

# INSTANCIO EL GENERADOR DE LA RESPUESTA (CONEXION OLLAMA-LLM)
generator = OllamaGenerator(
    model=MODEL_NAME1,
    url=OLLAMA_BASE_URL,
    timeout=450,
    generation_kwargs={
        "num_predict": 1000,
        "temperature": 0.5,
    },
)

# PIPELINE DE EJECUCION DEL CICLO DE CONSULTA/RESPUESTA
pipeline = Pipeline()
pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", generator)

pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")


# EJECUCION DEL PIPELINE CON PREGUNTA DE EJEMPLO
question = "Describe the Lighthouse of Alexandria, one of the Seven Wonders"

results = pipeline.run({
    "text_embedder": {"text": question},
    "prompt_builder": {"question": question}},
    include_outputs_from=["retriever"]
)


# DATA PARA DEBUGGEAR COMO FUNCIONA
print("LA RESPUESTA ES LA SIGUIENTE")
print(results["llm"]["replies"][0])

print("-----------------------------------")
print("LA RESPUESTA QUE SE OBTUVO DE ESTE CONTEXTO")
for doc in results["retriever"]["documents"]:
    print(doc.content)
    print(doc.meta)