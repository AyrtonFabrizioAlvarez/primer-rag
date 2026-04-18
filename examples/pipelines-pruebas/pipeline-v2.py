"""
title: Seven Wonders RAG (auto-ingest)
author: diuca
description: Pipeline RAG autocontenido. Indexa el dataset en pgvector la primera
             vez que se usa, y luego responde consultas. No requiere ejecutar
             indexacion.py previamente.
requirements: haystack-ai, ollama-haystack, datasets>=2.6.1, pgvector-haystack, python-dotenv
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import threading


class Pipeline:
    class Valves(BaseModel):
        # Conexión e infra
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        # Modelos
        EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
        GENERATION_MODEL: str = os.getenv("RAG_GENERATION_MODEL", "mistral:7b")

        # Dataset
        DATASET_NAME: str = os.getenv("RAG_DATASET_NAME", "bilgeyucel/seven-wonders")
        DATASET_SPLIT: str = os.getenv("RAG_DATASET_SPLIT", "train")

        # Vector store
        TABLE_NAME: str = os.getenv("RAG_TABLE_NAME", "haystack_docs_prueba")
        EMBEDDING_DIMENSION: int = int(os.getenv("RAG_EMBEDDING_DIM", "768"))

        # Retrieval
        TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))

        # Comportamiento
        FORCE_REINDEX: bool = os.getenv("RAG_FORCE_REINDEX", "false").lower() == "true"
        SHOW_SOURCES: bool = os.getenv("RAG_SHOW_SOURCES", "true").lower() == "true"

    def __init__(self):
        self.type = "manifold"
        self.id = "rag"
        self.name = "RAG: "
        self.pipelines = [{"id": "seven-wonders", "name": "Seven Wonders (Ollama)"}]
        self.valves = self.Valves()

        # Estado interno
        self._ready = False
        self._init_error = None
        self._rag_pipeline = None
        self._document_store = None
        # Lock para evitar ingestas concurrentes si llegan varias preguntas a la vez
        self._init_lock = threading.Lock()

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        # Si el usuario cambia valves desde la UI, invalidamos todo
        with self._init_lock:
            self._ready = False
            self._init_error = None
            self._rag_pipeline = None
            self._document_store = None

    # ---------------------------------------------------------------
    # INGESTA AUTOMÁTICA
    # ---------------------------------------------------------------
    def _ingest_if_needed(self, document_store) -> None:
        """
        Indexa el dataset en pgvector si la tabla está vacía
        o si FORCE_REINDEX=True. Si ya hay documentos, no hace nada.
        """
        from haystack import Pipeline as HaystackPipeline, Document
        from haystack.components.writers import DocumentWriter
        from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
        from datasets import load_dataset

        # Cuántos docs hay ya en la tabla
        try:
            doc_count = document_store.count_documents()
        except Exception as e:
            print(f"[RAG] No se pudo contar documentos (asumo 0): {e}")
            doc_count = 0

        if doc_count > 0 and not self.valves.FORCE_REINDEX:
            print(
                f"[RAG] Ingesta omitida: la tabla '{self.valves.TABLE_NAME}' "
                f"ya tiene {doc_count} documentos."
            )
            return

        print(
            f"[RAG] Ingestando dataset '{self.valves.DATASET_NAME}' "
            f"(split='{self.valves.DATASET_SPLIT}') en '{self.valves.TABLE_NAME}'..."
        )

        dataset = load_dataset(self.valves.DATASET_NAME, split=self.valves.DATASET_SPLIT)
        documents = [
            Document(content=d["content"], meta=d["meta"])
            for d in dataset
        ]
        print(f"[RAG] {len(documents)} documentos cargados desde el dataset.")

        document_embedder = OllamaDocumentEmbedder(
            model=self.valves.EMBEDDING_MODEL,
            url=self.valves.OLLAMA_BASE_URL,
            batch_size=32,
        )
        writer = DocumentWriter(document_store=document_store)

        index_pipeline = HaystackPipeline()
        index_pipeline.add_component("embedder", document_embedder)
        index_pipeline.add_component("writer", writer)
        index_pipeline.connect("embedder.documents", "writer.documents")

        index_pipeline.run({"embedder": {"documents": documents}})

        final_count = document_store.count_documents()
        print(f"[RAG] Ingesta completa. Total en la tabla: {final_count} documentos.")

    # ---------------------------------------------------------------
    # CONSTRUCCIÓN DEL PIPELINE DE CONSULTA
    # ---------------------------------------------------------------
    def _ensure_ready(self) -> None:
        if self._ready:
            return

        # Evita que varias requests concurrentes disparen la ingesta a la vez
        with self._init_lock:
            if self._ready:
                return

            try:
                from haystack import Pipeline as HaystackPipeline
                from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
                from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
                from haystack.components.builders import PromptBuilder
                from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
                from haystack_integrations.components.generators.ollama import OllamaGenerator

                # Conexión al vector store.
                # recreate_table=True SOLO si forzamos reindex, para no borrar datos existentes.
                document_store = PgvectorDocumentStore(
                    table_name=self.valves.TABLE_NAME,
                    embedding_dimension=self.valves.EMBEDDING_DIMENSION,
                    vector_function="cosine_similarity",
                    search_strategy="hnsw",
                    recreate_table=self.valves.FORCE_REINDEX,
                )
                self._document_store = document_store

                # AUTO-INGESTA
                self._ingest_if_needed(document_store)

                # Componentes de consulta
                text_embedder = OllamaTextEmbedder(
                    model=self.valves.EMBEDDING_MODEL,
                    url=self.valves.OLLAMA_BASE_URL,
                )

                retriever = PgvectorEmbeddingRetriever(
                    document_store=document_store,
                    top_k=self.valves.TOP_K,
                )

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
                # Después de una reindexación forzada, apagamos el flag en memoria
                # para que una eventual reinicialización no vuelva a tirar la tabla.
                if self.valves.FORCE_REINDEX:
                    self.valves.FORCE_REINDEX = False

            except Exception as e:
                self._init_error = str(e)
                self._ready = False

    # ---------------------------------------------------------------
    # ENTRYPOINT DE OPENWEBUI
    # ---------------------------------------------------------------
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        self._ensure_ready()

        if self._init_error:
            return (
                "Error inicializando Seven Wonders RAG: "
                f"{self._init_error}. "
                "Revisá que Ollama tenga los modelos descargados, que pgvector esté accesible "
                "y que PG_CONN_STR esté configurada en el contenedor de pipelines."
            )

        question = (user_message or "").strip()
        if not question:
            return "Escribí una pregunta para consultar el RAG de Seven Wonders."

        try:
            result = self._rag_pipeline.run(
                {
                    "text_embedder": {"text": question},
                    "prompt_builder": {"question": question},
                },
                include_outputs_from=["retriever"],
            )

            answer = result["llm"]["replies"][0]

            if not self.valves.SHOW_SOURCES:
                return answer

            docs = result["retriever"]["documents"]
            debug_context = "\n\n--- SOURCES ---\n"
            for d in docs:
                debug_context += f"\n{d.meta}\n{d.content[:300]}...\n"

            return answer + debug_context

        except Exception as e:
            return f"Error ejecutando Seven Wonders RAG: {e}"