from .azure_llm import AzureOpenAiLLM
from .fake_llm import FakeLLM
from .openai_llm import OpenAiLLM
from .llm import LLMRegistry, LlmConfig, LLM, LLMStreamController
from .exceptions import LlmError, LLMToolNotFound
from .types import ToolChoice, ErrorMessage
from .utils.completion_parsing import PromptParsingException

LLMRegistry.register(OpenAiLLM, default_model="gpt-4o")
# LLM.register(FakeLLM, default_model="pirate_stream.json")



__all__ = [
    # "PhiLLM",
    "LLM", 
    "LLMRegistry", 
    "LlmConfig",
    "LlmError", 
    "LLMToolNotFound", 
    "PromptParsingException", 
    "ToolChoice", 
    "ErrorMessage",
    "LLMStreamController",
]