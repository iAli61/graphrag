import asyncio
from collections.abc import Callable
from typing import Any, Sequence
from enum import Enum
import numpy as np
import tiktoken
import aiohttp
import requests
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.base import OpenAILLMImpl
from graphrag.query.llm.text_utils import chunk_text
from graphrag.query.progress import StatusReporter

class ApiType(str, Enum):
    """The API Flavor."""
    OpenAI = "openai"
    AzureOpenAI = "azure"
    Ollama = "ollama"
    vLLM = "vllm"
    SentenceTransformer = "sentence_transformer"

class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
    """Wrapper for various embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        azure_ad_token_provider: Callable | None = None,
        model: str = "text-embedding-3-small",
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: ApiType = ApiType.OpenAI,
        organization: str | None = None,
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8191,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
        reporter: StatusReporter | None = None,
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # type: ignore
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )

        self.model = model
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.retry_error_types = retry_error_types
        self.api_type = api_type

        if api_type == ApiType.Ollama:
            import ollama
            self.ollama_client = ollama.Client(host=api_base or "http://localhost:11434")
        elif api_type == ApiType.vLLM:
            self.vllm_base_url = api_base or "http://localhost:8000/v1/chat/completions"
        elif api_type == ApiType.SentenceTransformer:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model)

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        if self.api_type in [ApiType.OpenAI, ApiType.AzureOpenAI]:
            return self._openai_embed(text, **kwargs)
        elif self.api_type == ApiType.Ollama:
            return self._ollama_embed(text, **kwargs)
        elif self.api_type == ApiType.vLLM:
            return self._vllm_embed(text, **kwargs)
        elif self.api_type == ApiType.SentenceTransformer:
            return self._st_embed(text, **kwargs)
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        if self.api_type in [ApiType.OpenAI, ApiType.AzureOpenAI]:
            return await self._openai_aembed(text, **kwargs)
        elif self.api_type == ApiType.Ollama:
            return await self._ollama_aembed(text, **kwargs)
        elif self.api_type == ApiType.vLLM:
            return await self._vllm_aembed(text, **kwargs)
        elif self.api_type == ApiType.SentenceTransformer:
            return self._st_embed(text, **kwargs)  # SentenceTransformer doesn't have async API
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

    def _openai_embed(self, text: str, **kwargs: Any) -> list[float]:
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        for chunk in token_chunks:
            try:
                embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            except Exception as e:
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )
                continue
        if not chunk_embeddings:
            return []
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    async def _openai_aembed(self, text: str, **kwargs: Any) -> list[float]:
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        embedding_results = await asyncio.gather(*[
            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        ])
        embedding_results = [result for result in embedding_results if result[0]]
        if not embedding_results:
            return []
        chunk_embeddings = [result[0] for result in embedding_results]
        chunk_lens = [result[1] for result in embedding_results]
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    def _ollama_embed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            embedding = self.ollama_client.embeddings(model=self.model, prompt=text)
            return list(embedding["embedding"])
        except Exception as e:
            self._reporter.error(
                message="Error embedding text with Ollama",
                details={self.__class__.__name__: str(e)},
            )
            return []

    async def _ollama_aembed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            # Run the synchronous method in an executor
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                None, 
                self.ollama_client.embeddings, 
                self.model, 
                text
            )
            return list(embedding["embedding"])
        except Exception as e:
            self._reporter.error(
                message="Error embedding text asynchronously with Ollama",
                details={self.__class__.__name__: str(e)},
            )
            return []
    
    def _vllm_embed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            response = requests.post(
                self.vllm_base_url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": text}]
                }
            )
            response.raise_for_status()
            return list(response.json()["embedding"])
        except Exception as e:
            self._reporter.error(
                message="Error embedding text with vLLM",
                details={self.__class__.__name__: str(e)},
            )
            return []

    async def _vllm_aembed(self, text: str, **kwargs: Any) -> list[float]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.vllm_base_url,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": text}]
                    }
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return list(result["embedding"])
            except Exception as e:
                self._reporter.error(
                    message="Error embedding text asynchronously with vLLM",
                    details={self.__class__.__name__: str(e)},
                )
                return []

    def _st_embed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            embedding = self.st_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self._reporter.error(
                message="Error embedding text with SentenceTransformer",
                details={self.__class__.__name__: str(e)},
            )
            return []

    def _embed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    embedding = (
                        self.sync_client.embeddings.create(  # type: ignore
                            input=text,
                            model=self.model,
                            **kwargs,  # type: ignore
                        )
                        .data[0]
                        .embedding
                    )
                    return (list(embedding), len(text) if isinstance(text, str) else 0)
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
        return ([], 0)  # Return empty list and 0 length in case of any error

    async def _aembed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    embedding = (
                        await self.async_client.embeddings.create(  # type: ignore
                            input=text,
                            model=self.model,
                            **kwargs,  # type: ignore
                        )
                    ).data[0].embedding
                    return (list(embedding), len(text) if isinstance(text, str) else 0)
        except RetryError as e:
            self._reporter.error(
                message="Error at aembed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
        return ([], 0)  # Return empty list and 0 length in case of any error