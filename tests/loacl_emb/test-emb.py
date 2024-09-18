import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import yaml

from graphrag.llm.openai.embedding_llm import OpenAIEmbeddingsLLM, EmbeddingsLLMConfiguration, EmbeddingProviderType
from graphrag.llm.openai.openai_configuration import OpenAIConfiguration

# Set the default fixture loop scope to function
pytestmark = pytest.mark.asyncio(scope="function")

@pytest.fixture
def config():
    with open('settings.yaml', 'r') as file:
        return yaml.safe_load(file)

@pytest.fixture
async def mock_openai_client():
    client = AsyncMock()
    client.embeddings.create.return_value.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    return client

@pytest.fixture
def mock_ollama():
    with patch('graphrag.llm.openai.openai_embeddings_llm.ollama') as mock:
        mock.embeddings.return_value = {"embedding": [0.4, 0.5, 0.6]}
        yield mock

@pytest.fixture
def mock_sentence_transformer():
    with patch('graphrag.llm.openai.openai_embeddings_llm.SentenceTransformer', create=True) as mock:
        instance = mock.return_value
        instance.encode.return_value = np.array([[0.7, 0.8, 0.9]])
        yield instance

@pytest.mark.parametrize("provider", ["azure", "openai"])
async def test_openai_embeddings_llm_openai_or_azure(mock_openai_client, config, provider):
    if provider not in config:
        pytest.skip(f"{provider} configuration not found in settings.yaml")
    
    provider_config = config[provider]
    llm_config = EmbeddingsLLMConfiguration({
        "provider": provider,
        "api_key": provider_config['api_key'],
        "model": provider_config['model'],
        "api_base": provider_config.get('api_base'),
        "api_version": provider_config.get('api_version')
    })
    llm = OpenAIEmbeddingsLLM(mock_openai_client, llm_config)

    result = await llm._execute_llm(["Test input"])

    assert result == [[0.1, 0.2, 0.3]]
    mock_openai_client.embeddings.create.assert_called_once_with(input=["Test input"], model=provider_config['model'])

async def test_openai_embeddings_llm_ollama(mock_ollama, config):
    if 'ollama' not in config:
        pytest.skip("Ollama configuration not found in settings.yaml")
    
    ollama_config = config['ollama']
    llm_config = EmbeddingsLLMConfiguration({
        "provider": "ollama",
        "model": ollama_config['model']
    })
    llm = OpenAIEmbeddingsLLM(None, llm_config)

    result = await llm._execute_llm(["Test input"])

    assert result == [[0.4, 0.5, 0.6]]
    mock_ollama.embeddings.assert_called_once_with(model=ollama_config['model'], prompt="Test input")

async def test_openai_embeddings_llm_sentence_transformer(mock_sentence_transformer, config):
    if 'sentence_transformer' not in config:
        pytest.skip("SentenceTransformer configuration not found in settings.yaml")
    
    st_config = config['sentence_transformer']
    llm_config = EmbeddingsLLMConfiguration({
        "provider": "sentence_transformer",
        "model": st_config['model']
    })
    llm = OpenAIEmbeddingsLLM(mock_sentence_transformer, llm_config)

    result = await llm._execute_llm(["Test input"])

    assert result == [[0.7, 0.8, 0.9]]
    mock_sentence_transformer.encode.assert_called_once_with(["Test input"], convert_to_tensor=False)

async def test_openai_embeddings_llm_vllm(config):
    if 'vllm' not in config:
        pytest.skip("vLLM configuration not found in settings.yaml")
    
    vllm_config = config['vllm']
    llm_config = EmbeddingsLLMConfiguration({
        "provider": "vllm",
        "api_base": vllm_config['api_base'],
        "api_key": vllm_config['api_key'],
        "model": vllm_config['model']
    })
    llm = OpenAIEmbeddingsLLM(None, llm_config)

    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response

        result = await llm._execute_llm(["Test input"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_post.assert_called_once_with(
            f"{vllm_config['api_base']}/v1/embeddings",
            json={"model": vllm_config['model'], "input": ["Test input"]},
            headers={"Authorization": f"Bearer {vllm_config['api_key']}"}
        )

def test_openai_embeddings_llm_unsupported_provider():
    with pytest.raises(ValueError):
        EmbeddingsLLMConfiguration({
            "provider": "unsupported",
            "model": "test-model"
        })