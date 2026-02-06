import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, validator

from ...llms.messages import AIMessage, HumanMessage, SystemMessage
from .....legacy2.text.llms.rag_manager import RagValue, RagVector, RagVectorSpace
from langsmith import Client
# from ...common.vectors.embeddings.text_embeddings import DenseEmbeddings
from ..embeddings.text_embeddings import DenseEmbeddings




class ConversationRagMetadata(BaseModel):
    """inputs can be a list of messages or a single message"""
    # inputs: List[HumanMessage | AIMessage | SystemMessage]
    inputs: List[HumanMessage | AIMessage | SystemMessage]
    output: AIMessage




class RagConversationVectorizer:
    """
        vectorizer for conversation
    """

    def __init__(self) -> None:
        self.dense_embeddings = DenseEmbeddings()
    

    def _stringify_message(self, message: Union[AIMessage, HumanMessage, SystemMessage]):
        if type(message) == AIMessage:
            if message.tool_calls:
                return f"{message.role}: {message.content}\n{message.tool_calls} tool calls"
        return f"{message.role}: {message.content}"
    
    def _stringify_messages_list(self, messages):
        if type(messages) == list:
            return "\n".join([self._stringify_message(m) for m in messages])
        else:
            return self._stringify_message(messages)
        

    def _preprocess_key(self, value):
        if hasattr(value, "key"):
            return value.key
        else:
            return value
    
    async def _embed(self, docs: List[str]):
        return await asyncio.to_thread(self.dense_embeddings.embed_documents, docs)
    
    
    async def embed_documents(self, documents: List[ConversationRagMetadata]) -> List[RagVector]:
        embeddings = await self._embed([self._stringify_messages_list(doc.inputs) for doc in documents])
        # vectors = [RagVector[ConversationRagValue](key=e, value=v[0], id=v[1]) for e,v in  zip(embeddings, value_conversations)]
        return embeddings
    
    
    async def embed_query(self, query: List[HumanMessage | AIMessage | SystemMessage]):
        embd = await self._embed([self._stringify_messages_list(query)])
        return embd[0]
        