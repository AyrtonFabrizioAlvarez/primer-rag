"""
title: Seven Wonders RAG (auto-ingest | Ollama + Groq)
author: diuca
description: Pipeline RAG autocontenido con soporte dual de proveedor LLM.
             Embeddings siempre vía Ollama (local). Generación configurable:
             Ollama (local) o Groq (API gratuita con modelos grandes).
             La primera vez que se usa, indexa el dataset automáticamente en pgvector.
requirements: haystack-ai, ollama-haystack, datasets>=2.6.1, pgvector-haystack, openai
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import threading


class Pipeline:

    class Valves(BaseModel):
        # --- Infraestructura Ollama (siempre necesario para embeddings) ---
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

        # --- Proveedor de generación ---
        LLM_PROVIDER: str = os.getenv("RAG_LLM_PROVIDER", "ollama")  # "ollama" | "groq"

        # --- Modelos ---
        EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
        # Modelo Ollama (solo aplica si LLM_PROVIDER=ollama)
        OLLAMA_GENERATION_MODEL: str = os.getenv("RAG_GENERATION_MODEL", "mistral:7b")
        # Modelo Groq (solo aplica si LLM_PROVIDER=groq)
        GROQ_MODEL: str = os.getenv("RAG_GROQ_MODEL", "llama-3.3-70b-versatile")

        # --- Credenciales Groq ---
        GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        # --- Dataset ---
        DATASET_NAME: str = os.getenv("RAG_DATASET_NAME", "bilgeyucel/seven-wonders")
        DATASET_SPLIT: str = os.getenv("RAG_DATASET_SPLIT", "train")

        # --- Vector store ---
        TABLE_NAME: str = os.getenv("RAG_TABLE_NAME", "haystack_docs_prueba")
        EMBEDDING_DIMENSION: int = int(os.getenv("RAG_EMBEDDING_DIM", "768"))

        # --- Retrieval ---
        TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))

        # --- Comportamiento ---
        FORCE_REINDEX: bool = os.getenv("RAG_FORCE_REINDEX", "false").lower() == "true"
        SHOW_SOURCES: bool = os.getenv("RAG_SHOW_SOURCES", "true").lower() == "true"

    def __init__(self):
        self.type = "manifold"
        self.id = "rag"
        self.name = "RAG: "
        self.pipelines = [{"id": "seven-wonders", "name": "Seven Wonders"}]
        self.valves = self.Valves()

        self._ready = False
        self._init_error = None
        self._rag_pipeline = None
        self._document_store = None
        self._active_provider = None  # guarda cuál proveedor quedó activo
        self._init_lock = threading.Lock()

    async def on_startup(self):
        provider = self.valves.LLM_PROVIDER
        print(f"on_startup:{__name__} | proveedor configurado: {provider}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        """Al cambiar valves desde la UI de OpenWebUI, fuerza reinicialización."""
        with self._init_lock:
            self._ready = False
            self._init_error = None
            self._rag_pipeline = None
            self._document_store = None
            self._active_provider = None

    # ---------------------------------------------------------------
    # INGESTA AUTOMÁTICA
    # ---------------------------------------------------------------
    def _ingest_if_needed(self, document_store) -> None:
        """
        Indexa el dataset en pgvector solo si la tabla está vacía
        o si FORCE_REINDEX=True. Los embeddings siempre son vía Ollama.
        """
        from haystack import Pipeline as HaystackPipeline, Document
        from haystack.components.writers import DocumentWriter
        from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
        from datasets import load_dataset

        try:
            doc_count = document_store.count_documents()
        except Exception as e:
            print(f"[RAG] No se pudo contar documentos (asumo 0): {e}")
            doc_count = 0

        if doc_count > 0 and not self.valves.FORCE_REINDEX:
            print(
                f"[RAG] Ingesta omitida: '{self.valves.TABLE_NAME}' "
                f"ya tiene {doc_count} documentos."
            )
            return

        print(
            f"[RAG] Ingestando '{self.valves.DATASET_NAME}' "
            f"(split='{self.valves.DATASET_SPLIT}') → tabla '{self.valves.TABLE_NAME}'..."
        )
        dataset = load_dataset(self.valves.DATASET_NAME, split=self.valves.DATASET_SPLIT)
        documents = [
            Document(content=d["content"], meta=d["meta"])
            for d in dataset
        ]
        print(f"[RAG] {len(documents)} documentos cargados.")

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

        print(
            f"[RAG] Ingesta completa. "
            f"Total en tabla: {document_store.count_documents()} documentos."
        )

    # ---------------------------------------------------------------
    # CONSTRUCCIÓN DEL GENERADOR (Ollama o Groq)
    # ---------------------------------------------------------------
    def _build_generator(self):
        provider = self.valves.LLM_PROVIDER.lower().strip()

        if provider == "groq":
            if not self.valves.GROQ_API_KEY:
                raise ValueError(
                    "LLM_PROVIDER=groq pero GROQ_API_KEY no está definida. "
                    "Agregala en pipelines.env o en las valves del pipeline."
                )
            from haystack.components.generators import OpenAIGenerator
            from haystack.utils import Secret

            print(f"[RAG] Generador: Groq → modelo '{self.valves.GROQ_MODEL}'")
            return OpenAIGenerator(
                api_key=Secret.from_token(self.valves.GROQ_API_KEY),
                model=self.valves.GROQ_MODEL,
                api_base_url=self.valves.GROQ_BASE_URL,
                generation_kwargs={
                    "temperature": 0.5,
                    "max_tokens": 1000,
                },
            )

        else:  # ollama (default)
            from haystack_integrations.components.generators.ollama import OllamaGenerator

            print(
                f"[RAG] Generador: Ollama → modelo '{self.valves.OLLAMA_GENERATION_MODEL}' "
                f"en {self.valves.OLLAMA_BASE_URL}"
            )
            return OllamaGenerator(
                model=self.valves.OLLAMA_GENERATION_MODEL,
                url=self.valves.OLLAMA_BASE_URL,
                generation_kwargs={
                    "num_predict": 1000,
                    "temperature": 0.5,
                },
                timeout=450,
            )

    # ---------------------------------------------------------------
    # INICIALIZACIÓN COMPLETA DEL PIPELINE
    # ---------------------------------------------------------------
    def _ensure_ready(self) -> None:
        if self._ready:
            return

        with self._init_lock:
            if self._ready:
                return

            try:
                from haystack import Pipeline as HaystackPipeline
                from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
                from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
                from haystack.components.builders import PromptBuilder
                from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

                # Vector store
                document_store = PgvectorDocumentStore(
                    table_name=self.valves.TABLE_NAME,
                    embedding_dimension=self.valves.EMBEDDING_DIMENSION,
                    vector_function="cosine_similarity",
                    search_strategy="hnsw",
                    recreate_table=self.valves.FORCE_REINDEX,
                )
                self._document_store = document_store

                # Auto-ingesta si hace falta
                self._ingest_if_needed(document_store)

                # Embedder para queries (siempre Ollama)
                text_embedder = OllamaTextEmbedder(
                    model=self.valves.EMBEDDING_MODEL,
                    url=self.valves.OLLAMA_BASE_URL,
                )

                # Retriever
                retriever = PgvectorEmbeddingRetriever(
                    document_store=document_store,
                    top_k=self.valves.TOP_K,
                )

                # Prompt
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

                # Generador (Ollama o Groq según config)
                generator = self._build_generator()
                self._active_provider = self.valves.LLM_PROVIDER.lower()

                # Ensamblado del pipeline
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
            proveedor = self.valves.LLM_PROVIDER
            if proveedor == "groq":
                hint = (
                    "Revisá que GROQ_API_KEY esté definida en pipelines.env "
                    "y que el modelo en RAG_GROQ_MODEL sea válido "
                    "(ej: llama-3.3-70b-versatile, llama-3.1-8b-instant)."
                )
            else:
                hint = (
                    "Revisá que Ollama tenga los modelos descargados, "
                    "que pgvector esté accesible y que PG_CONN_STR esté configurada."
                )
            return f"Error inicializando RAG ({proveedor}): {self._init_error}. {hint}"

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
            provider_label = (
                f"Groq / {self.valves.GROQ_MODEL}"
                if self._active_provider == "groq"
                else f"Ollama / {self.valves.OLLAMA_GENERATION_MODEL}"
            )
            sources = f"\n\n--- SOURCES (via {provider_label}) ---\n"
            for d in docs:
                sources += f"\n{d.meta}\n{d.content[:300]}...\n"

            return answer + sources

        except Exception as e:
            return f"Error ejecutando RAG: {e}"