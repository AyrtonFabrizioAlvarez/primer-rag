from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os

class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
        GENERATION_MODEL: str = os.getenv("RAG_GENERATION_MODEL", "mistral")
        DATASET_NAME: str = os.getenv("RAG_DATASET_NAME", "bilgeyucel/seven-wonders")
        DATASET_SPLIT: str = os.getenv("RAG_DATASET_SPLIT", "train")
        TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))

    def __init__(self):
        self.type = "manifold"
        self.id = "rag"
        self.name = "RAG: "
        self.pipelines = [{"id": "seven-wonders", "name": "Seven Wonders (Ollama)"}]
        self.valves = self.Valves()

        self._ready = False
        self._init_error = None
        self._rag_pipeline = None

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        self._ready = False
        self._init_error = None
        self._rag_pipeline = None

    def _ensure_ready(self):
        if self._ready:
            return

        try:
            from haystack import Pipeline as HaystackPipeline, Document
            from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
            from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
            from haystack.components.builders import PromptBuilder
            from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
            from haystack_integrations.components.generators.ollama import OllamaGenerator

            document_store = PgvectorDocumentStore(
                table_name="haystack_docs_prueba",
                embedding_dimension=768,
                vector_function="cosine_similarity",
                search_strategy="hnsw",
            )


            text_embedder = OllamaTextEmbedder(
                model=self.valves.EMBEDDING_MODEL,
                url=self.valves.OLLAMA_BASE_URL,
            )

            retriever = PgvectorEmbeddingRetriever(document_store=document_store, top_k=2)


            template = """
                        Given the following information, answer the question.

                        Context:
                        {% for document in documents %}
                        {{ document.content }}
                        {% endfor %}

                        Question: {{question}}
                        Answer:
                    """
            prompt_builder = PromptBuilder(template=template)

            generator = OllamaGenerator(
                model=self.valves.GENERATION_MODEL,
                url=self.valves.OLLAMA_BASE_URL,
                generation_kwargs={
                    "num_predict": 1000,
                    "temperature": 0.5,
                },
                timeout=450,
            )

            rag_pipeline = HaystackPipeline()
            rag_pipeline.add_component("text_embedder", text_embedder)
            rag_pipeline.add_component("retriever", retriever)
            rag_pipeline.add_component("prompt_builder", prompt_builder)
            rag_pipeline.add_component("llm", generator)

            rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
            rag_pipeline.connect("retriever", "prompt_builder.documents")
            rag_pipeline.connect("prompt_builder", "llm")

            self._rag_pipeline = rag_pipeline
            self._ready = True
        except Exception as e:
            self._init_error = str(e)
            self._ready = False

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        self._ensure_ready()

        if self._init_error:
            return (
                "Error inicializando Seven Wonders RAG: "
                f"{self._init_error}. "
                "Revisa que Ollama tenga los modelos y que el contenedor pipelines pueda instalar dependencias."
            )

        question = (user_message or "").strip()
        if not question:
            return "Escribe una pregunta para consultar el RAG de Seven Wonders."

        try:
            result = self._rag_pipeline.run(
                {
                    "text_embedder": {"text": question},
                    "prompt_builder": {"question": question},
                },
                include_outputs_from=["retriever"]
            )
            #return result["llm"]["replies"][0]

            answer = result["llm"]["replies"][0]
            docs = result["retriever"]["documents"]

            debug_context = "\n\n--- SOURCES ---\n"
            for d in docs:
                debug_context += f"\n{d.meta}\n{d.content[:300]}...\n"

            return answer + debug_context

        except Exception as e:
            return f"Error ejecutando Seven Wonders RAG: {e}"
