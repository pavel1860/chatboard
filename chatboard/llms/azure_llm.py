
from typing import Optional

from .clients.azure_client import AzureOpenAiLlmClient
from .clients.base import BaseLlmClient
from .llm import LLM


class AzureOpenAiLLM(LLM):
    name: str = "AzureOpenAiLLM"
    client: BaseLlmClient
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_version: Optional[str] = "2023-12-01-preview"
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    

    def __init__(self, api_key, api_version, azure_endpoint, azure_deployment, **kwargs):
        client = AzureOpenAiLlmClient(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,            
        )
        super().__init__(
            client= client,            
            **kwargs
        )
