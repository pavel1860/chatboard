from typing import List

from promptview.llms.clients.phi_llm_client import PhiLlmClient
from promptview.llms.legacy.llm import LLM
from pydantic import Field


class PhiLLM(LLM):
    name: str = "PhiLLM"
    stop_sequences: List[str]=["Instruct"]
    client: PhiLlmClient = Field(default_factory=lambda: PhiLlmClient)
    model: str = "microsoft/phi-2"

