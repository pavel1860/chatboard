import os

import openai


def build_async_openai_client(api_key: str | None=None, api_version: str | None=None, azure_endpoint: str | None=None, azure_deployment: str | None=None):
    azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")    
    if azure_endpoint:        
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version=api_version or os.getenv("OPENAI_API_VERSION", "2023-12-01-preview")
        azure_deployment=azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        return openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    azure_deployment=azure_deployment,
                ) 
    elif os.environ.get("OPENAI_API_KEY", None):
        return openai.AsyncClient(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError("OpenAI API Key not found in environment variables")
    


def build_async_openai_embeddings_client(api_key: str | None=None, api_version: str | None=None, azure_endpoint: str | None=None, azure_deployment: str | None=None):
    azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")    
    if azure_endpoint:        
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        api_version=api_version or os.getenv("OPENAI_API_VERSION", "2023-12-01-preview")
        azure_deployment=azure_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        return openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    azure_deployment=azure_deployment,
                ) 
    elif os.environ.get("OPENAI_API_KEY", None):
        return openai.AsyncClient(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError("OpenAI API Key not found in environment variables")