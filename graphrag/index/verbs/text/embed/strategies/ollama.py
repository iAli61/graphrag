# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run method definition for Ollama embeddings."""

import asyncio
import logging
from typing import Any

from datashaper import VerbCallbacks, progress_ticker

from graphrag.index.cache import PipelineCache
from graphrag.index.utils import is_null
from graphrag.llm import OpenAIConfiguration  # Consider creating OllamaConfiguration
from graphrag.llm.openai.embedding_llm import load_embedding_llm, TextEmbedStrategyType

from .llm import get_splitter, execute_embedding, create_text_batches, prepare_embed_texts, reconstitute_embeddings
from .typing import TextEmbeddingResult

log = logging.getLogger(__name__)

async def run(
    input: list[str],
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    args: dict[str, Any],
) -> TextEmbeddingResult:
    """Run the Ollama embedding process."""
    if is_null(input):
        return TextEmbeddingResult(embeddings=None)

    llm_config = args.get("llm", {})
    batch_size = args.get("batch_size", 16)
    batch_max_tokens = args.get("batch_max_tokens", 8191)
    ollama_config = OpenAIConfiguration(llm_config)  # Consider creating OllamaConfiguration
    splitter = get_splitter(ollama_config, batch_max_tokens)
    llm = load_embedding_llm(TextEmbedStrategyType.ollama, ollama_config.raw_config)
    semaphore: asyncio.Semaphore = asyncio.Semaphore(args.get("num_threads", 4))

    try:
        # Break up the input texts. The sizes here indicate how many snippets are in each input text
        texts, input_sizes = prepare_embed_texts(input, splitter)
        text_batches = create_text_batches(
            texts,
            batch_size,
            batch_max_tokens,
            splitter,
        )
        log.info(
            "Ollama: embedding %d inputs via %d snippets using %d batches. max_batch_size=%d, max_tokens=%d",
            len(input),
            len(texts),
            len(text_batches),
            batch_size,
            batch_max_tokens,
        )
        ticker = progress_ticker(callbacks.progress, len(text_batches))

        # Embed each chunk of snippets
        embeddings = await execute_embedding(llm, text_batches, ticker, semaphore)
        embeddings = reconstitute_embeddings(embeddings, input_sizes)

        return TextEmbeddingResult(embeddings=embeddings)
    except Exception as e:
        log.error(f"Error in Ollama embedding process: {str(e)}")
        # Handle Ollama-specific errors here
        raise