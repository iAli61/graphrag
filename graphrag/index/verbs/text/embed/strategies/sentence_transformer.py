# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run method definition for SentenceTransformer embeddings."""

import asyncio
import logging
from typing import Any

import numpy as np
from datashaper import ProgressTicker, VerbCallbacks, progress_ticker

import graphrag.config.defaults as defs
from graphrag.index.cache import PipelineCache
from graphrag.index.text_splitting import TokenTextSplitter
from graphrag.index.utils import is_null
from graphrag.llm import OpenAIConfiguration
from graphrag.llm.openai.embedding_llm import load_embedding_llm, TextEmbedStrategyType, BaseEmbeddingsLLM

from .typing import TextEmbeddingResult

log = logging.getLogger(__name__)

async def run(
    input: list[str],
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    args: dict[str, Any],
) -> TextEmbeddingResult:
    """Run the SentenceTransformer embedding process."""
    if is_null(input):
        return TextEmbeddingResult(embeddings=None)

    llm_config = args.get("llm", {})
    batch_size = args.get("batch_size", 32)  # SentenceTransformer can typically handle larger batches
    st_config = OpenAIConfiguration(llm_config)  # We're reusing OpenAIConfiguration for simplicity
    splitter = _get_splitter(st_config, batch_size)
    llm = _get_llm(st_config)

    # SentenceTransformer doesn't require text splitting, but we'll keep the structure for consistency
    texts = input
    text_batches = _create_text_batches(texts, batch_size)

    log.info(
        "embedding %d inputs using %d batches. batch_size=%d",
        len(input),
        len(text_batches),
        batch_size,
    )
    ticker = progress_ticker(callbacks.progress, len(text_batches))

    # Embed each batch of texts
    embeddings = await _execute(llm, text_batches, ticker)

    return TextEmbeddingResult(embeddings=embeddings)

def _get_splitter(
    config: OpenAIConfiguration, batch_size: int
) -> TokenTextSplitter:
    # SentenceTransformer doesn't require text splitting, but we'll keep this for consistency
    return TokenTextSplitter(
        encoding_name=config.encoding_model or defs.ENCODING_MODEL,
        chunk_size=batch_size,
    )

def _get_llm(config: OpenAIConfiguration) -> BaseEmbeddingsLLM:
    return load_embedding_llm(TextEmbedStrategyType.sentence_transformer, config.raw_config)

async def _execute(
    llm: BaseEmbeddingsLLM,
    chunks: list[list[str]],
    tick: ProgressTicker,
) -> list[list[float]]:
    async def embed(chunk: list[str]):
        # SentenceTransformer is not async, so we'll run it in an executor
        loop = asyncio.get_running_loop()
        chunk_embeddings = await loop.run_in_executor(None, llm._execute_llm, chunk)
        result = np.array(chunk_embeddings)
        tick(1)
        return result

    futures = [embed(chunk) for chunk in chunks]
    results = await asyncio.gather(*futures)
    # merge results in a single list of lists
    return [item for sublist in results for item in sublist]

def _create_text_batches(
    texts: list[str],
    batch_size: int,
) -> list[list[str]]:
    """Create batches of texts to embed."""
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]