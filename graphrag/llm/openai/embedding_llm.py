# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Module for different embedding LLM implementations and strategy loading."""

from enum import Enum
from typing import Any, Union
from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import EmbeddingInput, EmbeddingOutput, LLMInput
from graphrag.config.enums import LLMType
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes

class TextEmbedStrategyType(str, Enum):
    """TextEmbedStrategyType class definition."""
    openai = "openai_embedding"
    azure = "azure_openai_embedding"
    ollama = "ollama_embedding"
    vllm = "vllm_embedding"
    sentence_transformer = "sentence_transformer_embedding"
    mock = "mock"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'

class BaseEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """Base class for embedding LLMs."""
    
    def __init__(self, configuration: Any):
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIEmbeddingsLLM(BaseEmbeddingsLLM):
    """OpenAI and Azure OpenAI embedding LLM."""

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        super().__init__(configuration)
        self.client = client

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        embedding = await self.client.embeddings.create(
            input=input,
            **args,
        )
        return [d.embedding for d in embedding.data]

class OllamaEmbeddingsLLM(BaseEmbeddingsLLM):
    """Ollama embedding LLM."""

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        import ollama
        embeddings = []
        for text in input:
            embedding = ollama.embeddings(
                model=self.configuration.model,
                prompt=text
            )
            embeddings.append(embedding["embedding"])
        return embeddings

class vLLMEmbeddingsLLM(BaseEmbeddingsLLM):
    """vLLM embedding LLM."""

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.configuration.api_base}/v1/embeddings",
                json={
                    "model": self.configuration.model,
                    "input": input,
                },
                headers={"Authorization": f"Bearer {self.configuration.api_key}"}
            ) as response:
                result = await response.json()
                return [d["embedding"] for d in result["data"]]

class SentenceTransformerEmbeddingsLLM(BaseEmbeddingsLLM):
    """SentenceTransformer embedding LLM."""

    def __init__(self, configuration: Any):
        super().__init__(configuration)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.configuration.model)

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        embeddings = self.model.encode(input, convert_to_tensor=False)
        return embeddings.tolist()

def load_embedding_llm(strategy: TextEmbedStrategyType, config: dict, client: Any = None) -> BaseEmbeddingsLLM:
    """Load the appropriate embedding LLM based on the strategy type."""
    if strategy in [TextEmbedStrategyType.openai, TextEmbedStrategyType.azure]:
        return OpenAIEmbeddingsLLM(client, OpenAIConfiguration(config))
    elif strategy == TextEmbedStrategyType.ollama:
        return OllamaEmbeddingsLLM(config)
    elif strategy == TextEmbedStrategyType.vllm:
        return vLLMEmbeddingsLLM(config)
    elif strategy == TextEmbedStrategyType.sentence_transformer:
        return SentenceTransformerEmbeddingsLLM(config)
    else:
        raise ValueError(f"Unsupported embedding strategy: {strategy}")

def load_strategy(strategy: Union[str, TextEmbedStrategyType]) -> Any:
    """Load strategy method definition."""
    if isinstance(strategy, str):
        strategy = TextEmbedStrategyType(strategy)
    
    match strategy:
        case TextEmbedStrategyType.openai | TextEmbedStrategyType.azure:
            from graphrag.index.verbs.text.embed.strategies.openai import run as run_openai
            return run_openai
        case TextEmbedStrategyType.ollama:
            from graphrag.index.verbs.text.embed.strategies.ollama import run as run_ollama
            return run_ollama
        case TextEmbedStrategyType.vllm:
            from graphrag.index.verbs.text.embed.strategies.vllm import run as run_vllm
            return run_vllm
        case TextEmbedStrategyType.sentence_transformer:
            from graphrag.index.verbs.text.embed.strategies.sentence_transformer import run as run_sentence_transformer
            return run_sentence_transformer
        case TextEmbedStrategyType.mock:
            from graphrag.index.verbs.text.embed.strategies.mock import run as run_mock
            return run_mock
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)