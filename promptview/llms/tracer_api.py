from datetime import datetime
from typing import Dict, List, Union

from openai.types.chat.chat_completion_message_tool_call import \
    ChatCompletionMessageToolCall
from ..llms.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel


def get_message(run_data):
    # print(run_data)
    if run_data['role'] == 'user':
        return HumanMessage(content=run_data['content'])
    if run_data['role'] == 'assistant':
        tool_calls = None
        # if run_data['additional_kwargs'].get('tool_calls'):
        if run_data.get('tool_calls'):
            tool_calls = [ChatCompletionMessageToolCall(**tc) for tc in run_data['additional_kwargs']['tool_calls']]            
        if isinstance(run_data['content'], str): 
            return AIMessage(content=run_data['content'], tool_calls=tool_calls)
        else:
            return AIMessage(content='', content_blocks=run_data['content'], tool_calls=tool_calls)
    if run_data['role'] == 'system':
        return SystemMessage(content=run_data['content'])
    else:
        raise ValueError(f"Unknown role: {run_data['role']}")


class RetrieverDocument(BaseModel):
    id: str
    score: float
    content: str | None = None


class LsRunBase(BaseModel):
    id: str
    name: str
    run_type: str
    inputs: Dict = {}
    metadata: Dict = {}
    output: Dict | str = {}
    start_time: datetime
    end_time: datetime
    error: str | None = None


    
class LsLlmRun(LsRunBase):
    messages: list = []
    
    def __getitem__(self, key):
        return self.messages[key]
    
    def __repr__(self) -> str:
        return f"\t{self.name} - {self.run_type}" + "\n" + "\n".join([f"\t{i}. {m.role}" for i, m in enumerate(self.messages)])
    
    def show(self, idx=0, tabs=0):
        # return f"{'\t' * tabs}{self.name} - {self.run_type}\n" + "\n".join([f"'\t\t\t'{i}. {m.role}" for i, m in enumerate(self.messages)])
        tabs_str = '\t' * tabs
        tabs_msg_str = '\t' * (tabs + 2)
        return f"{tabs_str}{idx}. {self.name} - {self.run_type}\n" + "\n".join([f"{tabs_msg_str}{i}. {m.role}" for i, m in enumerate(self.messages)])


class RetrieverRun(LsRunBase):
    documents: List[RetrieverDocument] = []


class LsRun(LsRunBase):
    children: List[Union['LsRun', 'LsLlmRun', 'RetrieverRun']] = []
    
    def __getitem__(self, key):
        return self.children[key]
        
    def get_names(self):
        return [c.name for c in self.children]
    
    @property
    def messages(self):
        if self.children and self.children[0].run_type == "llm":
            return self.children[0].messages # type: ignore
        return None
    
    @property
    def input_message(self):
        messages = self.messages
        if messages:
            return messages[-2]
    
    @property
    def output_message(self):
        messages = self.messages
        if messages:
            return messages[-1]
    
    def __repr__(self) -> str:
        return self.show()
    
    def show(self, idx=None, tabs=0 ):
        idx_str = f"{idx}. " if idx is not None else ""
        tabs_str = '\t' * tabs
        return f"{tabs_str}{idx_str}{self.name} - {self.run_type}\n" + "\n".join([f"{c.show(i, tabs+1)}" for i, c in enumerate(self.children)]) # type: ignore
    
    def __iter__(self):
        return iter(self.children)

    

def get_run_messages(run):
    if run.run_type == "llm":
        # print(run.name)
        ls_llm_run = LsLlmRun(
            id=str(run.id), 
            name=run.name, 
            run_type=run.run_type,
            start_time=run.start_time,
            end_time=run.end_time,
            metadata=run.metadata,
            error=run.error
        )
        messages = [get_message(m) for m in run.inputs['messages']]
        # print("----output----")
        messages.append(get_message(run.outputs))            
        ls_llm_run.messages = messages
        return ls_llm_run    
    elif run.run_type == "retriever":
        documents = run.outputs.get('documents', [])
        ls_doc_run = RetrieverRun(
            id=str(run.id), 
            name=run.name, 
            run_type=run.run_type,
            start_time=run.start_time,
            end_time=run.end_time,
            inputs=run.inputs,
            output=run.outputs,
            metadata=run.metadata,
            documents=[RetrieverDocument(
                id=doc['metadata']['document_id'], 
                score=doc['metadata']['score'],
                content=doc['page_content']
            ) for doc in documents],
            error=run.error
        )
        return ls_doc_run
    
    ls_run = LsRun(
            id=str(run.id), 
            name=run.name, 
            run_type=run.run_type, 
            inputs=run.inputs.get('input', {}),
            output=run.outputs.get('output', {}) if run.outputs else {},
            metadata=run.metadata,
            start_time=run.start_time,
            end_time=run.end_time,
            error=run.error
        )
    if run.child_runs:
        for child_run in run.child_runs:
            ls_child = get_run_messages(child_run)
            ls_run.children.append(ls_child)
    return ls_run

    
