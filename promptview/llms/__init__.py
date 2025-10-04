from .azure_llm import AzureOpenAiLLM
from .fake_llm import FakeLLM
# from .openai_llm2 import OpenAiLLM
from .openai_llm3 import OpenAiLLM
# from .legacy.phi_llm import PhiLLM
# from .llm import LLM, OutputModel
from .llm import OutputModel
from .llm2 import LLM, LlmConfig
from .exceptions import LlmError, LLMToolNotFound
from .types import ToolChoice, ErrorMessage
from .utils.completion_parsing import PromptParsingException

LLM.register(OpenAiLLM, default_model="gpt-4o")
# LLM.register(FakeLLM, default_model="pirate_stream.json")



__all__ = [
    # "PhiLLM", 
    "LLM", 
    "LlmConfig",
    "LlmError", 
    "LLMToolNotFound", 
    "PromptParsingException", 
    "ToolChoice", 
    "ErrorMessage",
    "OutputModel",
]