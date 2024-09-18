# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing OpenAI-specific embedding implementation."""

import asyncio
import logging
from typing import Any

from datashaper import VerbCallbacks, progress_ticker

import graphrag.config.defaults as defs
from graphrag.index.cache import PipelineCache
from graphrag.index.text_splitting import TokenTextSplitter
from graphrag.index.utils import is_null
from graphrag.llm import OpenAIConfiguration

from .typing import TextEmbeddingResult
from .llm import get_llm, execute_embedding, create_text_batches, prepare_embed_texts, reconstitute_embeddings

log = logging.getLogger(__name__)

async def run(
    input: list[str],
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    args: dict[str, Any],
) -> TextEmbeddingResult:
    """Run the OpenAI embedding process."""
    if is_null(input):
        return TextEmbeddingResult(embeddings=None)

    llm_config = args.get("llm", {})
    batch_size = args.get("batch_size", 16)
    batch_max_tokens = args.get("batch_max_tokens", 8191)
    oai_config = OpenAIConfiguration(llm_config)
    splitter = _get_splitter(oai_config, batch_max_tokens)
    llm = get_llm(oai_config, callbacks, cache)
    semaphore: asyncio.Semaphore = asyncio.Semaphore(args.get("num_threads", 4))

    # Break up the input texts. The sizes here indicate how many snippets are in each input text
    texts, input_sizes = prepare_embed_texts(input, splitter)
    text_batches = create_text_batches(
        texts,
        batch_size,
        batch_max_tokens,
        splitter,
    )
    log.info(
        "embedding %d inputs via %d snippets using %d batches. max_batch_size=%d, max_tokens=%d",
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

def _get_splitter(
    config: OpenAIConfiguration, batch_max_tokens: int
) -> TokenTextSplitter:
    return TokenTextSplitter(
        encoding_name=config.encoding_model or defs.ENCODING_MODEL,
        chunk_size=batch_max_tokens,
    )
