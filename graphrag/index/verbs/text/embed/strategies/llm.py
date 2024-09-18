# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing generic LLM-related functionality."""

import asyncio
import logging
from typing import Any

import numpy as np
from datashaper import ProgressTicker, VerbCallbacks, progress_ticker

from graphrag.index.cache import PipelineCache
from graphrag.index.text_splitting import TokenTextSplitter
from graphrag.index.utils import is_null
from graphrag.llm import EmbeddingLLM, OpenAIConfiguration
from graphrag.llm.openai.embedding_llm import load_embedding_llm, TextEmbedStrategyType, BaseEmbeddingsLLM
import graphrag.config.defaults as defs


log = logging.getLogger(__name__)

def get_llm(config: OpenAIConfiguration, callbacks: VerbCallbacks, cache: PipelineCache) -> BaseEmbeddingsLLM:
    llm_type_str = config.lookup("type", "openai")
    
    try:
        strategy_type = TextEmbedStrategyType(llm_type_str)
    except ValueError:
        # Fallback to OpenAI if the type is not recognized
        strategy_type = TextEmbedStrategyType.openai
        log.warning(f"Unrecognized embedding strategy type: {llm_type_str}. Falling back to OpenAI.")

    return load_embedding_llm(strategy_type, config.raw_config)

def get_splitter(
    config: OpenAIConfiguration, batch_max_tokens: int
) -> TokenTextSplitter:
    return TokenTextSplitter(
        encoding_name=config.encoding_model or defs.ENCODING_MODEL,
        chunk_size=batch_max_tokens,
    )

async def execute_embedding(
    llm: EmbeddingLLM,
    chunks: list[list[str]],
    tick: ProgressTicker,
    semaphore: asyncio.Semaphore,
) -> list[list[float]]:
    async def embed(chunk: list[str]):
        async with semaphore:
            chunk_embeddings = await llm(chunk)
            result = np.array(chunk_embeddings.output)
            tick(1)
        return result

    futures = [embed(chunk) for chunk in chunks]
    results = await asyncio.gather(*futures)
    # merge results in a single list of lists (reduce the collect dimension)
    return [item for sublist in results for item in sublist]

def create_text_batches(
    texts: list[str],
    max_batch_size: int,
    max_batch_tokens: int,
    splitter: TokenTextSplitter,
) -> list[list[str]]:
    """Create batches of texts to embed."""
    result = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts:
        token_count = splitter.num_tokens(text)
        if (
            len(current_batch) >= max_batch_size
            or current_batch_tokens + token_count > max_batch_tokens
        ):
            result.append(current_batch)
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += token_count

    if len(current_batch) > 0:
        result.append(current_batch)

    return result

def prepare_embed_texts(
    input: list[str], splitter: TokenTextSplitter
) -> tuple[list[str], list[int]]:
    sizes: list[int] = []
    snippets: list[str] = []

    for text in input:
        # Split the input text and filter out any empty content
        split_texts = splitter.split_text(text)
        if split_texts is None:
            continue
        split_texts = [text for text in split_texts if len(text) > 0]

        sizes.append(len(split_texts))
        snippets.extend(split_texts)

    return snippets, sizes

def reconstitute_embeddings(
    raw_embeddings: list[list[float]], sizes: list[int]
) -> list[list[float] | None]:
    """Reconstitute the embeddings into the original input texts."""
    embeddings: list[list[float] | None] = []
    cursor = 0
    for size in sizes:
        if size == 0:
            embeddings.append(None)
        elif size == 1:
            embedding = raw_embeddings[cursor]
            embeddings.append(embedding)
            cursor += 1
        else:
            chunk = raw_embeddings[cursor : cursor + size]
            average = np.average(chunk, axis=0)
            normalized = average / np.linalg.norm(average)
            embeddings.append(normalized.tolist())
            cursor += size
    return embeddings
